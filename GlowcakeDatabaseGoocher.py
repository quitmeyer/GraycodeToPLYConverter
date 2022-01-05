# Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

# This script is based on an original implementation by True Price.

from sqlite3.dbapi2 import Cursor, connect
import sys
import sqlite3
import numpy as np
import csv

IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tobytes()
    else:
        return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)


    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(self, model, width, height, params,
                   prior_focal_length=False, camera_id=None):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, array_to_blob(params),
             prior_focal_length))
        return cursor.lastrowid

    def add_image(self, name, camera_id,
                  prior_q=np.full(4, np.NaN), prior_t=np.full(3, np.NaN), image_id=None):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, camera_id, prior_q[0], prior_q[1], prior_q[2],
             prior_q[3], prior_t[0], prior_t[1], prior_t[2]))
        return cursor.lastrowid

    def add_keypoints(self, image_id, keypoints):
        assert(len(keypoints.shape) == 2)
        assert(keypoints.shape[1] in [2, 4, 6])

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),))

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),))

    def add_matches(self, image_id1, image_id2, matches):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),))

    def update_two_view_geometries(cursor, matches, pair_id):
        cursor.execute("UPDATE two_view_geometries SET data=?, rows=?, cols=? WHERE pair_id=?",
            (array_to_blob(matches), matches.shape[0], matches.shape[1], pair_id))


    def add_two_view_geometry(self, image_id1, image_id2, matches,
                              F=np.eye(3), E=np.eye(3), H=np.eye(3),
                              qvec=np.array([1.0, 0.0, 0.0, 0.0]),
                              tvec=np.zeros(3), config=2):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        qvec = np.asarray(qvec, dtype=np.float64)
        tvec = np.asarray(tvec, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches), config,
             array_to_blob(F), array_to_blob(E), array_to_blob(H),
             array_to_blob(qvec), array_to_blob(tvec)))

    


def example_usage():
    import os
    import argparse
    print("doing stuff")

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="database.db")
    parser.add_argument("--camAPoints", default="ProjPointsCamA.CSV")
    parser.add_argument("--camBPoints", default="ProjPointsCamB.CSV")


    args = parser.parse_args()

    if os.path.exists(args.db):
        print("Good! database path already exists -- .")
        print("let's fuck it up")
    #    return

    if os.path.exists(args.camAPoints):
        print("Good! cam A points exist! let's read em! -- .")
        print(args.camAPoints)
    #    return

    
    # Open the database.

    db = COLMAPDatabase.connect(args.db)
    print("database opened")

    # For convenience, try creating all the tables upfront.

    db.create_tables()
    '''
    AllImages = db.execute("SELECT * from Images")
    for i in AllImages:
        print("AllImages  ")  
        print(i)   

    AllCameras = db.execute("SELECT * from Cameras")
    for i in AllCameras:
        print("AllCameras  ")  
        print(i)  
    '''

    ### Get the Canonical Camera's ID
    print("\n List Canonical Camera A  ")  

    canonCamA = db.execute("SELECT * from Images where name='a/camA_WB_1.png'")
 

    for i in canonCamA:
        print("\n canonical Camera A  ")  
        print(i)  
        print("\n Canon Cam A Image ID ")  
        print(i[0])
        camA_image_id = i[0]


    print("\n List Canonical Camera B  ")  

    canonCamB = db.execute("SELECT * from Images where name='b/camB_WB_1.png'")
    for i in canonCamB:
        print("\n canonical Camera B  ")  
        print(i)  
        print("\n Canon Cam B Image ID ")  
        print(i[0])
        camB_image_id = i[0]

    


#~~~~ CREATE CAMERA OBJECT FOR PROJECTOR ~~~~~~~~~~~~~

    """
	float w = params.width; // projector w
			float h = params.height; // projector h
			float diagnonalFOV = dFOV;

			///Building a guess from our measured fields of view of the projectors

			//1920 x 1080  big projector 35.1	    0.6126105675
		   // 1366 x 768 small projector 72.88	1.271995959

			float d = sqrt(powf(w, 2) + powf(h, 2));
			float f = (d / 2) * cos(diagnonalFOV / 2) / sin(diagnonalFOV / 2);  // old guess  1.732; // 1.732 = cotangent(1.0472/2) where 1.0472 is 60 degrees in radians)

			cameraMatrixG.at< float >(0, 0) = f;
			cameraMatrixG.at< float >(1, 1) = f;
			cameraMatrixG.at< float >(0, 2) = w / 2; // assume it's about in the center
			cameraMatrixG.at< float >(1, 2) = h / 2; // assume it's about in the center

    """
    # Create dummy cameras.

    #model1, width1, height1, params1 = \
    #    0, 1024, 768, np.array((1024., 512., 384.))
    #model2, width2, height2, params2 = \
    #    2, 1024, 768, np.array((1024., 512., 384., 0.1))

    #create dummy camera for the projector
        # model 4 is OPENCV, 0 is simple pinhole  2 is simple radial
        #values taken from guess calculations
    p_model1, p_width1, p_height1, p_params1 =  4, 1366, 768, np.array((1.732,1.732, 1366/2, 768/2,0,0,0,0)) #you need this many arguments for an openCV param or it will crash everything

    p_camera_id1 = db.add_camera(p_model1, p_width1, p_height1, p_params1)
    print("added the projector camera with id = "+str(p_camera_id1))

    # Create Graycode Match images.
    
    proj_image_id = db.add_image("extras/small_projector_gradient_1366.png", p_camera_id1) #fake graycode from projector
    #camA_image_id = db.add_image("extras/CamB_WB_1.png", 1)     #view of Black image from graycode in CAMA
    #camB_image_id = db.add_image("extras/CamA_WB_1.png", 2)     #view of black image graycode in CAMB //Note the cameras are flipped


    #~~~~ READ GRAYCODE FROM CSV FILES ~~~~~~~~~~~~~

    #CAMERA A GRAYCODE CSV
    # initializing the titles and rows list
    fieldsA = []
    rowsA = []
    # reading csv file for Camera A
    with open(args.camAPoints, 'r') as csvfile:
    # creating a csv reader object
        csvreader = csv.reader(csvfile)
      
    # extracting field names through first row
        fieldsA = next(csvreader)
  
    # extracting each data row one by one
        for row in csvreader:
            rowsA.append(row)
  
    # get total number of rows
        print("Total no. of rows: %d"%(csvreader.line_num))
        totalRows = csvreader.line_num
    # printing the field names
    print('Field names are:' + ', '.join(field for field in fieldsA))

    '''
    #  printing first 5 rows
    print('\nFirst 5 rows are:\n')
    for row in rowsA[:5]:
        # parsing each column of a row
        # Field names are:Xc, Yc, Xp, Yp
        for col in row:
            print("%10s"%col),
        print('\n')
        #db.add_keypoints(image_id1, (int(lines[0]),int(lines[0])))
    '''
    
    #CAMERA B GRAYCODE CSV
    # initializing the titles and rows list
    fieldsB = []
    rowsB = []
    # reading csv file for Camera A
    with open(args.camBPoints, 'r') as csvfileB:
    # creating a csv reader object
        csvreaderB = csv.reader(csvfileB)
      
    # extracting field names through first row
        fieldsB = next(csvreaderB)
  
    # extracting each data row one by one
        for row in csvreaderB:
            rowsB.append(row)
  
    # get total number of rows
        print("Total no. of rows: %d"%(csvreaderB.line_num))
        totalRowsB = csvreaderB.line_num
    # printing the field names
    print('Field names are:' + ', '.join(field for field in fieldsB))
    '''
    #  printing first 5 rows
    print('\nFirst 5 rows of B are:\n')
    for row in rowsB[:5]:
        # parsing each column of a row
        # Field names are:Xc, Yc, Xp, Yp
        for col in row:
            print("%10s"%col),
        print('\n')
        #db.add_keypoints(image_id1, (int(lines[0]),int(lines[0])))
    '''

    #go through ALL rows
    #for row in rows:
        #keypoint = np.array([(int(row[0]),int(row[1]))])
        #print(row[0]+"  "+row[1])
        #print(keypoint)
        #n=n+1
    
    """  numpy.delete(arr, obj, axis=None)

    arr refers to the input array,
    obj refers to which sub-arrays (e.g. column/row no. or slice of the array) and
    axis refers to either column wise (axis = 1) or row-wise (axis = 0) delete operation.
    """


    #------------Add keypoints-------------------

    print("\n List Canonical Camera A Keypoints \n -----  ")  

    canonCamAKeypoints = db.execute("SELECT * from keypoints where image_id='"+str(camA_image_id)+"'")
    #for i in canonCamAKeypoints:
    #     print("\n canonical Camera A  ")  
         #print(i)
    image_id, rows, cols, keypointblob = next(canonCamAKeypoints)
    keypointarray = np.frombuffer(keypointblob, np.float32).reshape(rows, cols)

    print(keypointarray.shape)
    print("last index of orignal keypoints ")
    canCamKeypointsOrigIndex =keypointarray.shape[0] -1
    print(canCamKeypointsOrigIndex)
    print(keypointarray)

    print("Augmented keypoints ")
    #add the keypoints

    #Graycode Keypoints for Cam A
    cKeysA = np.delete(rowsA,np.s_[2:4],1)
    print("cKeysA ")
    print(cKeysA)
    print(cKeysA.shape)
    numofKeypointsAdded=cKeysA.shape[0]
    #empty_array = np.empty((cKeysA.shape[0], 4), np.float32)
    ones_array = np.ones((cKeysA.shape[0], 4), np.float32)
    
    cKeysA = np.append(cKeysA,ones_array,axis=1)
    print(cKeysA.shape)
    #augmentedKeypointArray = np.append(keypointarray,np.asarray([[1,2,3,4,5,6]],np.float32) ) #EXAMPLE
    augmentedKeypointArray = np.append(keypointarray,cKeysA)


    #reshape it back how the database likes it
    augmentedKeypointArray =augmentedKeypointArray.reshape(rows+numofKeypointsAdded,cols)
    print("Keypoints data array augmented")

    augmentedKeypointArray = np.asarray(augmentedKeypointArray, np.float32)
    print(augmentedKeypointArray.shape)

    #put those keypoints into that database!
    data_tuple = (camA_image_id, augmentedKeypointArray.shape[0], augmentedKeypointArray.shape[1],array_to_blob(augmentedKeypointArray))

    canonCamA = db.execute("INSERT or REPLACE INTO keypoints (image_id, rows, cols, data) VALUES (?, ?, ?, ?)",data_tuple)


    #print("cKeysA after Subtraction (for inverted Y's) ") #note the CSV's we get from the graycode machine can sometimes spit out Y coordinates reversed of what they should be for Colmap

    cam_width = 4096
    cam_height = 2160
    #cKeysA = -2160+cKeysA
    #print(cKeysA)
    
    #dummy test
    keypoints1 = np.random.rand(3, 2) * (100, 200)

    #db.add_keypoints(camA_image_id, keypoints1)

    # Add placeholder scale, orientation.
    #keypoints = np.concatenate([keypoints, np.ones((n_keypoints, 1)), np.zeros((n_keypoints, 1))], axis=1).astype(np.float32)

    #keypoints_str = keypoints1.tostring()
    #db.execute("INSERT INTO keypoints(image_id, rows, cols, data) VALUES(?, ?, ?, ?);",  (camA_image_id, keypoints1.shape[0], keypoints1.shape[1], keypoints_str))


    #Graycode Keypoints for Projector (Projector to CAMERA A GRAYCODE Correspondences)
    pKeysA=np.delete(rowsA,np.s_[0:2],1)
    print("\npKeysA ")
    print(pKeysA)

    db.add_keypoints(proj_image_id, pKeysA) #add to the projector


    #Graycode Keypoints for Cam B
    cKeysB = np.delete(rowsB,np.s_[2:4],1)
    print("cKeysB ")
    print(cKeysB)

    #db.add_keypoints(camB_image_id, cKeysB)


    pKeysB=np.delete(rowsB,np.s_[0:2],1)
    print("\npKeysB ")
    print(pKeysB)

    #db.add_keypoints(proj_image_id, pKeysB) #add to the projector from B  #gives errrrror

    '''
    #add matches
    print("add matches to A")

    #arr = np.array([[1, 2, 3], [4, 5, 6]])
    arrA = np.arange(0, totalRows, 1)
    #matches_CA_P = np.array(pKeysA, cKeysA)

    matches_CA_P = np.array([arrA,arrA])
    print(matches_CA_P.shape)
    print(matches_CA_P)
    matches_CA_P =matches_CA_P.T #have to transpose it
    print(matches_CA_P.shape)

    #db.add_matches(proj_image_id, camA_image_id, matches_CA_P)

    #db.update_two_view_geometries( matches_CA_P, image_ids_to_pair_id(proj_image_id,camA_image_id))

    print("NOTE ADDING RIGHT NOW_!add matches to B")
    arrB = np.arange(0, totalRowsB, 1)

    matches_CB_P = np.array([arrB,arrB])
    print(matches_CB_P.shape)
    print(matches_CB_P)
    matches_CB_P =matches_CB_P.T #have to transpose it
    print(matches_CB_P.shape)

    #db.add_matches(proj_image_id, camB_image_id, matches_CB_P) #incorrect method of adding matches right now
 
    # Create dummy keypoints.
    #
    # Note that COLMAP supports:
    #      - 2D keypoints: (x, y)
    #      - 4D keypoints: (x, y, theta, scale)
    #      - 6D affine keypoints: (x, y, a_11, a_12, a_21, a_22)
 
    """ num_keypoints = 1000
    keypoints1 = np.random.rand(num_keypoints, 2) * (p_width1, p_height1)
    keypoints2 = np.random.rand(num_keypoints, 2) * (p_width1, p_height1)
    keypoints3 = np.random.rand(num_keypoints, 2) * (p_width1, p_height1)
    keypoints4 = np.random.rand(num_keypoints, 2) * (p_width1, p_height1)

    db.add_keypoints(proj_image_id1, keypoints1)
    db.add_keypoints(image_id2, keypoints2)
    db.add_keypoints(image_id3, keypoints3) """



    # Create dummy matches.

    """    M = 50
    matches12 = np.random.randint(num_keypoints, size=(M, 2))
    matches23 = np.random.randint(num_keypoints, size=(M, 2))
    matches34 = np.random.randint(num_keypoints, size=(M, 2))

    db.add_matches(proj_image_id1, image_id2, matches12)
    db.add_matches(image_id2, image_id3, matches23) """

    #~~~~~ Commit the data to the file.
    '''
    
    db.commit()
    print("saved db")

    # Read and check cameras.
 
    rowsA = db.execute("SELECT * FROM cameras")
    camera_id, model, width, height, params, prior = next(rowsA)
    params = blob_to_array(params, np.float64)

    print(params)

    #assert camera_id == p_camera_id1
    #assert model == p_model1 and width == p_width1 and height == p_height1
    #assert np.allclose(params, p_params1)
    """
    camera_id, model, width, height, params, prior = next(rows)
    params = blob_to_array(params, np.float64)
    assert camera_id == camera_id2
    assert model == model2 and width == width2 and height == height2
    assert np.allclose(params, params2)

    # Read and check keypoints.

    keypoints = dict(
        (image_id, blob_to_array(data, np.float32, (-1, 2)))
        for image_id, data in db.execute(
            "SELECT image_id, data FROM keypoints"))

    assert np.allclose(keypoints[image_id1], keypoints1)
    assert np.allclose(keypoints[image_id2], keypoints2)
    assert np.allclose(keypoints[image_id3], keypoints3)
    assert np.allclose(keypoints[image_id4], keypoints4)

    # Read and check matches.

    pair_ids = [image_ids_to_pair_id(*pair) for pair in
                ((image_id1, image_id2),
                 (image_id2, image_id3),
                 (image_id3, image_id4))]

    matches = dict(
        (pair_id_to_image_ids(pair_id),
         blob_to_array(data, np.uint32, (-1, 2)))
        for pair_id, data in db.execute("SELECT pair_id, data FROM matches")
    )
    print("pair ids")
    print(pair_ids)
    assert np.all(matches[(image_id1, image_id2)] == matches12)
    assert np.all(matches[(image_id2, image_id3)] == matches23)
    assert np.all(matches[(image_id3, image_id4)] == matches34)
    """

    # Clean up.
    
    db.close()

    #if os.path.exists(args.database_path):
     #   os.remove(args.database_path)


if __name__ == "__main__":
    example_usage()
