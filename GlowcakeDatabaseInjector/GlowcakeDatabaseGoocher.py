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
import shutil
import pandas as pd

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
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches), config,
             array_to_blob(F), array_to_blob(E), array_to_blob(H)))

    


def example_usage():
    import os
    import argparse
    print("doing stuff")

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="database.db")
    parser.add_argument("--camAPoints", default="ProjPointsCamA.CSV")
    parser.add_argument("--camBPoints", default="ProjPointsCamB.CSV")


    args = parser.parse_args()

    shutil.copy(args.db,args.db+"new.db")

    if os.path.exists(args.db+"new.db"):
        print("Good! database path already exists -- .")
        print("let's fuck it up")
    #    return

    if os.path.exists(args.camAPoints):
        print("Good! cam A points exist! let's read em! -- .")
        print(args.camAPoints)
    #    return

    
    # Open the database.
    
    db = COLMAPDatabase.connect(args.db+"new.db")
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
    
    proj_image_id = db.add_image("projector/white1366.png", p_camera_id1) #fake graycode from projector
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
        print("Total no. of rows (Cam B): %d"%(csvreaderB.line_num))
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

    #--------------A-------------------
    #add keypoints for Camera A canon image
    print("\n List Canonical Camera A Keypoints \n -----  ")  

    canonCamAKeypoints = db.execute("SELECT * from keypoints where image_id='"+str(camA_image_id)+"'")
    #for i in canonCamAKeypoints:
    #     print("\n canonical Camera A  ")  
         #print(i)
    image_id, rows, cols, keypointblob = next(canonCamAKeypoints)
    keypointarrayA = np.frombuffer(keypointblob, np.float32).reshape(rows, cols)

    print(keypointarrayA.shape)
    print("last index of orignal keypoints ")
    canCamAKeypointsOrigIndex =keypointarrayA.shape[0] -1
    print(canCamAKeypointsOrigIndex)
    print(keypointarrayA)

    print("Augmented keypoints ")
    #add the keypoints

    #Graycode Keypoints for Cam A
    cKeysA = np.delete(rowsA,np.s_[2:4],1)
    print("cKeysA ")
    print(cKeysA)
    print(cKeysA.shape)
    numofKeypointsAddedA=cKeysA.shape[0]
    #empty_array = np.empty((cKeysA.shape[0], 4), np.float32)
    ones_array = np.ones((cKeysA.shape[0], 4), np.float32)
    
    cKeysA = np.append(cKeysA,ones_array,axis=1)
    print(cKeysA.shape)
    #augmentedKeypointArray = np.append(keypointarray,np.asarray([[1,2,3,4,5,6]],np.float32) ) #EXAMPLE
    augmentedKeypointArrayA = np.append(keypointarrayA,cKeysA)


    #reshape it back how the database likes it
    augmentedKeypointArrayA =augmentedKeypointArrayA.reshape(rows+numofKeypointsAddedA,cols)
    print("Keypoints data array augmented")

    augmentedKeypointArrayA = np.asarray(augmentedKeypointArrayA, np.float32)
    print(augmentedKeypointArrayA.shape)

    #put those keypoints into that database!
    data_tuple = (camA_image_id, augmentedKeypointArrayA.shape[0], augmentedKeypointArrayA.shape[1],array_to_blob(augmentedKeypointArrayA))

    canonCamA = db.execute("INSERT or REPLACE INTO keypoints (image_id, rows, cols, data) VALUES (?, ?, ?, ?)",data_tuple)


    #print("cKeysA after Subtraction (for inverted Y's) ") #note the CSV's we get from the graycode machine can sometimes spit out Y coordinates reversed of what they should be for Colmap

    cam_width = 4096
    cam_height = 2160
    #cKeysA = -2160+cKeysA
    #print(cKeysA)
    
  
    #db.add_keypoints(camA_image_id, keypoints1)

    # Add placeholder scale, orientation.
    #keypoints = np.concatenate([keypoints, np.ones((n_keypoints, 1)), np.zeros((n_keypoints, 1))], axis=1).astype(np.float32)

    #keypoints_str = keypoints1.tostring()
    #db.execute("INSERT INTO keypoints(image_id, rows, cols, data) VALUES(?, ?, ?, ?);",  (camA_image_id, keypoints1.shape[0], keypoints1.shape[1], keypoints_str))


    #---------B----------------
    #repeat the process for Cam B
    #Graycode Keypoints for Cam B
    #add keypoints for Camera A canon image
    print("\n List Canonical Camera B Keypoints \n -----  ")  

    #db.add_keypoints(camB_image_id, cKeysB)
    canonCamBKeypoints = db.execute("SELECT * from keypoints where image_id='"+str(camB_image_id)+"'")
    #for i in canonCamAKeypoints:
    #     print("\n canonical Camera A  ")  
         #print(i)
    image_idB, bRows, bCols, keypointblobB = next(canonCamBKeypoints)
    keypointarrayB = np.frombuffer(keypointblobB, np.float32).reshape(bRows, bCols)

    print(keypointarrayB.shape)
    print("last index of orignal keypoints ")
    canCamBKeypointsOrigIndex =keypointarrayB.shape[0] -1
    print(canCamBKeypointsOrigIndex)
    print(keypointarrayB)

    print("Augmented keypoints Cam B ")
    #add the keypoints

    #Graycode Keypoints for Cam B
    cKeysB = np.delete(rowsB,np.s_[2:4],1)
    print("cKeysB ")
    print(cKeysB)
    print(cKeysB.shape)
    numofKeypointsAddedB=cKeysB.shape[0]
    #empty_array = np.empty((cKeysA.shape[0], 4), np.float32)
    ones_arrayB = np.ones((cKeysB.shape[0], 4), np.float32)
    
    cKeysB = np.append(cKeysB,ones_arrayB,axis=1)
    print(cKeysB.shape)
    augmentedKeypointArrayB = np.append(keypointarrayB,cKeysB)


    #reshape it back how the database likes it
    augmentedKeypointArrayB =augmentedKeypointArrayB.reshape(bRows+numofKeypointsAddedB,bCols)
    print("Keypoints data array augmented")

    augmentedKeypointArrayB = np.asarray(augmentedKeypointArrayB, np.float32)
    print(augmentedKeypointArrayB.shape)

    #put those keypoints into that database!
    data_tupleB = (camB_image_id, augmentedKeypointArrayB.shape[0], augmentedKeypointArrayB.shape[1],array_to_blob(augmentedKeypointArrayB))

    canonCamB = db.execute("INSERT or REPLACE INTO keypoints (image_id, rows, cols, data) VALUES (?, ?, ?, ?)",data_tupleB)


    #--------------P------------------
    # adding keypoints for Projector camera
    print("~~ adding keypoints for the Projector ~~~ ")

    #Graycode Keypoints for Projector (Projector to CAMERA A GRAYCODE Correspondences)
    pKeysA=np.delete(rowsA,np.s_[0:2],1)
    print("\npKeysA ")
    print(pKeysA)

    #db.add_keypoints(proj_image_id, pKeysA) #add to the projector

    pKeysB=np.delete(rowsB,np.s_[0:2],1)
    print("\npKeysB ")
    print(pKeysB)
    
    pKeysA_pKeysB = np.vstack((pKeysA,pKeysB))  #have to use vstack and put in double parentheses for some weird reason

    print("\npKeysA and B ")
    print(pKeysA_pKeysB)

    db.add_keypoints(proj_image_id, pKeysA_pKeysB) #have to add both at once




    # -------- Add to matches and  two_view_geometries ----------

    # testing on how to get TWO VIEW GEOM into a variable
    '''
    camAcamBpairID = image_ids_to_pair_id(camA_image_id,camB_image_id)
    camAcamBTwoViewGeometries = db.execute("SELECT * from two_view_geometries where pair_id='"+str(camAcamBpairID)+"'")

    pair_id, rowsTVG, colsTVG, blobTVG, configTVG, F, E, H = next(camAcamBTwoViewGeometries)
    TVGarray = np.frombuffer(blobTVG, np.int32).reshape(rowsTVG, colsTVG)

    print(TVGarray.shape)
    print("TVGarray:")
    print(TVGarray)
   
    ProCamATVGarray = np.empty((0,TVGarray.shape[1]), int)
    #ProCamATVGarray = np.ones((3, TVGarray.shape[1]), np.int32) # rows and cols - cols will be 2
    ProCamATVGarray = np.append(ProCamATVGarray, np.array([[100,2000]]), axis=0)
    '''
    
    #CAM A - PROJ
    # create the indices of aligned matches between CAM A and PROJ
    arrA = np.arange(canCamAKeypointsOrigIndex, canCamAKeypointsOrigIndex+numofKeypointsAddedA, 1)
    arrP_A = np.arange(0,numofKeypointsAddedA, 1)

    matches_CA_P = np.array([arrA,arrP_A])

    print("matches_CA_P:")
    print(matches_CA_P.shape)
    print(matches_CA_P)
    matches_CA_P =matches_CA_P.T #have to transpose it
    print(matches_CA_P.shape)
    print(matches_CA_P)

    #db.add_matches(camA_image_id,proj_image_id,matches_CA_P)
    #db.update_two_view_geometries( matches_CA_P, image_ids_to_pair_id(camA_image_id,proj_image_id))
    db.add_two_view_geometry(camA_image_id,proj_image_id,matches_CA_P)

    #CAM B - PROJ
    # create the indices of aligned matches between CAM A and PROJ
    arrB = np.arange(canCamBKeypointsOrigIndex, canCamBKeypointsOrigIndex+numofKeypointsAddedB, 1)
    arrP_B = np.arange(numofKeypointsAddedA,numofKeypointsAddedA+ numofKeypointsAddedB, 1) #the keypoints for the projector for camB were appended to the projector's list of keypoints
    #NOTE possible off-by-one error above

    matches_CB_P = np.array([arrB,arrP_B])

    print("matches_CB_P:")
    print(matches_CB_P.shape)
    print(matches_CB_P)
    matches_CB_P =matches_CB_P.T #have to transpose it
    print(matches_CB_P.shape)
    print(matches_CB_P)


    db.add_two_view_geometry(camB_image_id,proj_image_id,matches_CB_P)


    #CAM A - CAM B
    # create the indices of aligned matches between CAM A and CAM B
    # replace with PANDAS
    print("Starting Intersection of two dataframes with PANDAS") 
    
    dA = pd.DataFrame(pKeysA) 
    therows, thecolumns = pKeysA.shape
    idx_a =np.arange(0, therows, 1)
    dA = dA.assign(idxA=idx_a)
    print("dataframeA: ")
    print(dA) 

    dB = pd.DataFrame(pKeysB)
    therows, thecolumns = pKeysB.shape
    idx_b =np.arange(0, therows, 1)
    dB = dB.assign(idxB=idx_b)
    print("dataframeB: ")
    print(dB) 

    # Calling merge() function 

    int_dfAtoB = pd.merge(dA, dB, how ='inner', on =[0, 1]) 
    int_dfBtoA = pd.merge(dA, dB, how ='inner', on =[0, 1]) 

    print("All Matched Rows between A to B")    
    print(int_dfAtoB)
    print("Matched rows between B to A") 
    print(int_dfBtoA)
    #print("Remove redundant rows") 
    #int_dfAtoB_noDups = int_dfAtoB.drop_duplicates()
    #print(int_dfAtoB_noDups)
    npAtoB = int_dfAtoB.to_numpy()
    print(npAtoB)
    idxs_A_B = np.delete(npAtoB,0,1) #kill first row
    idxs_A_B = np.delete(idxs_A_B,0,1) #kill the new first row
    print("these are the indices that correspond to matches from projA to projB")
    print(idxs_A_B)


    

    #TODO
    #then call bundle adjuster?
    #run the mapper in a special way   --Mapper.filter_max_reproj_error arg (=4)   (use special arguments )

    '''
    args.extract = False  # skip feature extraction
    args.match = False  # skip matching
    args.update_matches = True
    args.map = True
    args.update_two_view_geometries = True
    args.triangulate = True
    args.bundle_adjust = True
    args.num_threads = "1"
    args.ba_local_max_num_iterations = "30"
    args.ba_local_max_refinements = "3"
    args.ba_global_max_num_iterations = "75"
    args.min_num_matches = "1"
    args.ignore_two_view_tracks = "0"

    #call to colmap mapper
    def colmap_mapper(db_path, img_path, output_path, num_threads, ba_local_max_num_iterations, ba_local_max_refinements, ba_global_max_num_iterations):
    # parameters for mapper correspond to defaults for
    # high quality images when using automatic_reconstructor
    # num_threads helps remove non determinism due to ceres multi-threading
    cmd = 'colmap mapper' + \
        ' --database_path ' + db_path + \
        ' --image_path ' + img_path + \
        ' --output_path ' + output_path + \



    if args.map:
        copyfile(current_db_path, env.mapped_db)
        current_db_path = env.mapped_db
        colmap_mapper(current_db_path, env.images_path, env.sparse_model_path, args.num_threads,
                      args.ba_local_max_num_iterations, args.ba_local_max_refinements, args.ba_global_max_num_iterations)

    '''



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
