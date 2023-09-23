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

from pickle import FALSE, TRUE
from sqlite3.dbapi2 import Cursor, connect
import sys
import sqlite3
import numpy as np
import csv
import shutil
import pandas as pd
import random

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
                              tvec=np.zeros(3), config=2): #note andy changed this from 2
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
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?,?,?)",
            (pair_id,) + matches.shape + (array_to_blob(matches), config,
             array_to_blob(F), array_to_blob(E), array_to_blob(H),qvec, tvec)) #andy fixed this function by adding the qvec and tvec to the end

    


def example_usage():
    import os
    import argparse
    print("doing stuff")

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="database.db")
    parser.add_argument("--camAPoints", default="ProjPointsCamA.CSV")
    parser.add_argument("--camBPoints", default="ProjPointsCamB.CSV")
    parser.add_argument("--projWidth", default="1920")
    parser.add_argument("--projHeight", default="1080")
    parser.add_argument("--projImage", default="white1920.png")
    parser.add_argument("--subSample", default="1")
    parser.add_argument("--camModel", default="SimplePinhole")



    args = parser.parse_args()

    #rowsA = ONLYTAKE EVERY Xth entry
    subSample=int(args.subSample)
    newDBname = args.db[:-3]+"_"+args.camModel+"_"+str(subSample)+"_new.db"
    shutil.copy(args.db, newDBname)

    if os.path.exists(newDBname):
        print("Good! database path already exists -- .")
        print("let's fuck it up")
    #    return

    if os.path.exists(args.camAPoints):
        print("Good! cam A points exist! let's read em! -- .")
        print(args.camAPoints)
    #    return


    # Open the database.
    
    db = COLMAPDatabase.connect(newDBname)
    print("database opened")

    # For convenience, try creating all the tables upfront.

    db.create_tables()


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
           # 3840 2160 is big 4k projector

			float d = sqrt(powf(w, 2) + powf(h, 2));
			float f = (d / 2) * cos(diagnonalFOV / 2) / sin(diagnonalFOV / 2);  // old guess  1.732; // 1.732 = cotangent(1.0472/2) where 1.0472 is 60 degrees in radians)
			cameraMatrixG.at< float >(0, 0) = f;
			cameraMatrixG.at< float >(1, 1) = f;
			cameraMatrixG.at< float >(0, 2) = w / 2; // assume it's about in the center
			cameraMatrixG.at< float >(1, 2) = h / 2; // assume it's about in the center

    """

    #Create dummy camera for the projector
            # in colmap model 4 is OPENCV, 0 is simple pinhole  2 is simple radial
    """
             in colmap we could do " SIMPLE RADIAL model" which has

            " f, cx, cy, k  (If we just ignore k2 and k3)

            or RADIAL CAMERA model

            f, cx, cy, k1, k2 (if we ignore k3)


            there's also
            SIMPLE RADIAL FISHEYE
            f, cx, cy, k

            and that seems to be it of our stock blender options
    """

    match args.camModel:
        case "SimplePinhole":
                p_model1, p_width1, p_height1, p_params1 =  0, args.projWidth, args.projHeight, np.array((5400, int(args.projWidth)/2, int(args.projHeight)*.8)) #you need this many arguments for an SIMPLE PINHOLE param or it will crash everything
                p_camera_id1 = db.add_camera(p_model1, p_width1, p_height1, p_params1)
                print("added the "+args.camModel+" projector camera with id = "+str(p_camera_id1))

        case "SimpleRadial":
                p_model1, p_width1, p_height1, p_params1 =  2, args.projWidth, args.projHeight, np.array((5400, int(args.projWidth)/2, int(args.projHeight)*.8,0)) #you need this many arguments for an SIMPLE RADIAL param or it will crash everything
                p_camera_id1 = db.add_camera(p_model1, p_width1, p_height1, p_params1)
                print("added the "+args.camModel+" projector camera with id = "+str(p_camera_id1))
        case "OpenCV":
                p_model1, p_width1, p_height1, p_params1 =  4, args.projWidth, args.projHeight, np.array((5700, 5200, int(args.projWidth)/2, int(args.projHeight)*.8,0,0,0,0)) #you need this many arguments for an SIMPLE RADIAL param or it will crash everything
                p_camera_id1 = db.add_camera(p_model1, p_width1, p_height1, p_params1)
                print("added the "+args.camModel+" projector camera with id = "+str(p_camera_id1))

        #values taken from guess calculations
    #p_model1, p_width1, p_height1, p_params1 =  4, 1366, 768, np.array((1.732,1.732, 1366/2, 768/2,0,0,0,0)) #you need this many arguments for an openCV param or it will crash everything
    #p_model1, p_width1, p_height1, p_params1 =  4, 1920, 1080, np.array((1.732,1.732, 1920/2, 1080/2,0,0,0,0)) #you need this many arguments for an openCV param or it will crash everything
    #p_model1, p_width1, p_height1, p_params1 =  4, 3840, 2160, np.array((1.732,1.732, 3840/2, 2160/2,0,0,0,0)) #you need this many arguments for an openCV param or it will crash everything
    #opencv model
    #p_model1, p_width1, p_height1, p_params1 =  4, args.projWidth, args.projHeight, np.array((1.732,1.732, int(args.projWidth)/2, int(args.projHeight)/2,0,0,0,0)) #you need this many arguments for an openCV param or it will crash everything
    


    # Create the Image for the Projector
    proj_image_id = db.add_image("projector/"+args.projImage, p_camera_id1) #fake graycode from projector


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


    print("before and after reduction")
    print(len(rowsA))
    rowsAorig=rowsA.copy()
    if(subSample>0):
        rowsA = random.sample(rowsA,subSample)
    #rowsA=rowsA[0::skipinterval]
    print(len(rowsA))
   
   
   
    #CAMERA B GRAYCODE CSV
    # initializing the titles and rows list
    fieldsB = []
    rowsB = []
    # reading csv file for Camera B
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

    print("before and after reduction")
   
    print(len(rowsB))
    rowsBorig=rowsB.copy()
    if(subSample>0):
        rowsB = random.sample(rowsB,subSample)
    #rowsB=rowsB[0::skipinterval]
    print(len(rowsB))

#-------------Create Mega Database of only matches from the CSVs --- for right now, we don't care about graycode points that only Cam A saw or just CAM B, we want full triangle
    # # create the indices of aligned matches between CAM A and CAM B
    print("-----Creating Main Array of Matches from CSV files-------") 

    
    # RUN PANDAS for database matching
    print("Starting Intersection of two CSV files with PANDAS") 
    np_rowsA=np.asarray(rowsA)
    dA = pd.DataFrame(np_rowsA) 
    therows, thecolumns = np_rowsA.shape
    idx_a =np.arange(0, therows, 1)
    dA = dA.assign(idxA=idx_a)
    print("dataframeA: ")
    print(dA) 

    np_rowsB=np.asarray(rowsB)
    dB = pd.DataFrame(np_rowsB)
    therowsB, thecolumnsB = np_rowsB.shape
    idx_b =np.arange(0, therowsB, 1)
    dB = dB.assign(idxB=idx_b)
    print("dataframeB: ")
    print(dB) 


    # Calling pandas merge() function #We are only looking at places where the 
    int_dfAtoB = pd.merge(dA, dB, how ='inner', on =[2, 3]) 

    print("All Matched Rows between A to B")    
    print(int_dfAtoB)


    print("Remove duplicate indicies")  #there don't be any change when this is run  with JUST drop_duplicates
    # attempting to remove all in specific columns using subset command df.drop_duplicates(subset=[‘Color’])   
    int_dfAtoB_noDups = int_dfAtoB.drop_duplicates(subset=['idxA']) #drop duplicates of colA
    int_dfAtoB_noDups = int_dfAtoB_noDups.drop_duplicates(subset=['idxB']) #drop duplicates of colB

    print(int_dfAtoB_noDups)

    
    print("Pandas converted to numpy array")
    mainArray = int_dfAtoB_noDups.to_numpy()
    print("This is the main array that holds all our keypoints found in CamA Proj and CamB \n It's arranged as XcA YcA Xp Yp (index of original CSVA) XcB YcB (indx b) ")

    print(mainArray)

    print("Total amount of augmented keypoints")
    totalNewKeypoints = mainArray.shape[0]
    print(totalNewKeypoints)

    #------------Add keypoints-------------------

    #--------------A-------------------
    #add keypoints for Camera A canon image
    print("\n List Canonical Camera A Keypoints \n -----  ")  

    canonCamAKeypoints = db.execute("SELECT * from keypoints where image_id='"+str(camA_image_id)+"'")

    image_id, rows, cols, keypointblob = next(canonCamAKeypoints)
    keypointarrayA = np.frombuffer(keypointblob, np.float32).reshape(rows, cols)

    print(keypointarrayA.shape)
    print("last index of orignal keypoints ")
    canCamAKeypointsOrigIndex =keypointarrayA.shape[0] -1
    print(canCamAKeypointsOrigIndex)
    print(keypointarrayA)
    print(keypointarrayA[canCamAKeypointsOrigIndex])

    print("Augmented keypoints ")
    #add the keypoints

    #Graycode Keypoints for Cam A
    cKeysA = np.delete(mainArray,np.s_[2:8],1)
    print("cKeysA ")
    print(cKeysA)
    print(cKeysA.shape)

 

    numofKeypointsAddedA=cKeysA.shape[0]
    ones_array = np.ones((cKeysA.shape[0], 4), np.float32)
    
    cKeysA = np.append(cKeysA,ones_array,axis=1)
    print(cKeysA.shape)
    augmentedKeypointArrayA = np.append(keypointarrayA,cKeysA)


    #reshape it back how the database likes it
    augmentedKeypointArrayA =augmentedKeypointArrayA.reshape(rows+numofKeypointsAddedA,cols)
    print("Keypoints data array augmented")

    augmentedKeypointArrayA = np.asarray(augmentedKeypointArrayA, np.float32)
    print(augmentedKeypointArrayA.shape)

    #put those keypoints into that database!
    data_tuple = (camA_image_id, augmentedKeypointArrayA.shape[0], augmentedKeypointArrayA.shape[1],array_to_blob(augmentedKeypointArrayA))

    canonCamA = db.execute("INSERT or REPLACE INTO keypoints (image_id, rows, cols, data) VALUES (?, ?, ?, ?)",data_tuple)



    #---------B----------------
    #repeat the process for Cam B
    #Graycode Keypoints for Cam B
    #add keypoints for Camera A canon image
    print("\n List Canonical Camera B Keypoints \n -----  ")  

    canonCamBKeypoints = db.execute("SELECT * from keypoints where image_id='"+str(camB_image_id)+"'")

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
    cKeysB = np.delete(mainArray,np.s_[0:5],1)
    cKeysB = np.delete(cKeysB,np.s_[2:3],1)

    print("cKeysB ")
    print(cKeysB)
    print(cKeysB.shape)
    numofKeypointsAddedB=cKeysB.shape[0]
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
    pKeys=np.delete(mainArray,np.s_[4:8],1)
    pKeys=np.delete(pKeys,np.s_[0:2],1)

    print("\npKeysA ")
    print(pKeys)
    print(pKeys.shape)

    #pKeysA_pKeysB = np.vstack((pKeysA,pKeysB))  #have to use vstack and put in double parentheses for some weird reason
    db.add_keypoints(proj_image_id, pKeys)


    #------------ !Matches! --------------------------------------------
    # -------- Add Projector matches to matches and  two_view_geometries ----------

    print("-------Creating Matches between Proj and each cam-------") 

    #CAM A - PROJ
    # create the indices of aligned matches between CAM A and PROJ
    arrA = np.arange(canCamAKeypointsOrigIndex+1, canCamAKeypointsOrigIndex+1+totalNewKeypoints, 1)
    arrP_A = np.arange(0,totalNewKeypoints, 1)

    matches_A_P = np.array([arrA,arrP_A])

    print("matches_A_P:")
    print(matches_A_P.shape)
    print(matches_A_P)
    matches_A_P =matches_A_P.T #have to transpose it
    print(matches_A_P.shape)
    print(matches_A_P)

    db.add_matches(camA_image_id,proj_image_id,matches_A_P)    
    db.add_two_view_geometry(camA_image_id,proj_image_id,matches_A_P) 

    #CAM B - PROJ
    # create the indices of aligned matches between CAM B and PROJ
    arrB = np.arange(canCamBKeypointsOrigIndex+1, canCamBKeypointsOrigIndex+1+totalNewKeypoints, 1)
    arrP_B = np.arange(0, totalNewKeypoints, 1) #the keypoints for the projector for camB were appended to the projector's list of keypoints

    matches_B_P = np.array([arrB,arrP_B])

    print("matches_B_P:")
    print(matches_B_P.shape)
    print(matches_B_P)
    matches_B_P =matches_B_P.T #have to transpose it
    print(matches_B_P.shape)
    print(matches_B_P)
       
    db.add_matches(camB_image_id,proj_image_id,matches_B_P)
    db.add_two_view_geometry(camB_image_id,proj_image_id,matches_B_P) 



    print("-----Creating Matches between Cam A and Cam B-------") 
    #CAM A - Cam B - We should be able to add matches directly now, but these matches are a bit trickier because of how the database works with existing matches
    print("Adding Matches A to B")

    #Matches

    newMatches_A_B = np.array([arrA,arrB])
    newMatches_A_B =newMatches_A_B.T #have to transpose it


    #TVGPairs
    newTVGpairs_A_B = np.array([arrA,arrB])
    newTVGpairs_A_B = newTVGpairs_A_B.T
    print("matches with correct offset for the keypoints")
    print(newMatches_A_B)
    print(newMatches_A_B.shape)

    print("TVGpairs with correct offset for the keypoints")
    print(newTVGpairs_A_B)
    print(newTVGpairs_A_B.shape)

    #----------------------Inserting the Graycode Matches Canon A to B into the MATCHES database -----------
    print("\n List DB Matches \n -----  ")  
    
    ABpair_id = image_ids_to_pair_id(camA_image_id, camB_image_id)
    print(ABpair_id)
    canonCamA_B_MatchesOrig = db.execute("SELECT * from matches where pair_id='"+str(ABpair_id)+"'")

    thepair_id_Matches, rowsM, colsM, MatchesdataBlob  = next(canonCamA_B_MatchesOrig)
    Matches_arrayA_B_orig = np.frombuffer(MatchesdataBlob, np.uint32).reshape(rowsM, colsM)
    
    print("pair id for canon cameras images")
    print(thepair_id_Matches) #should match the one above and does
    print("original matches Cam A to cam b")
    print(Matches_arrayA_B_orig)
    print(Matches_arrayA_B_orig.shape)

    print("last index of orignal Matches ")
    Matches_OrigIndexA_B =Matches_arrayA_B_orig.shape[0] -1
    print(Matches_OrigIndexA_B)

    print("_------------------ADDING -----Augmented Matches ---Hopefully ")

    numofMATCHESAddedA=totalNewKeypoints
    
    #QUICKHACK
    augmentedMatchesArrayAB = np.append(Matches_arrayA_B_orig, newMatches_A_B)
    print(numofMATCHESAddedA)
    #reshape it back how the database likes it
    augmentedMatchesArrayAB =augmentedMatchesArrayAB.reshape(rowsM+numofMATCHESAddedA,colsM)
 
    
    print("Matches data array augmented")
    print(augmentedMatchesArrayAB.shape)

    augmentedMatchesArrayAB = np.asarray(augmentedMatchesArrayAB, np.uint32)
    print(augmentedMatchesArrayAB.shape)
    print(augmentedMatchesArrayAB)

    #put those matches into that database!
    #data_tuple = (ABpair_id, augmentedMatchesArrayAB.shape[0], augmentedMatchesArrayAB.shape[1],array_to_blob(augmentedMatchesArrayAB)) #this method also works
    #canonCamA_B_Matches = db.execute("INSERT or REPLACE INTO matches (pair_id, rows, cols, data) VALUES (?, ?, ?, ?)",data_tuple)
    
    #add to Matches DB
    canonCamA_B_MatchesOrig = db.execute("INSERT or REPLACE INTO matches VALUES (?,?,?,?)",(ABpair_id,)+augmentedMatchesArrayAB.shape+(array_to_blob(augmentedMatchesArrayAB),)) 
    print("matches added to database A-B")
    #db.add_matches(camA_image_id, camB_image_id, augmentedMatchesArrayAB) # can only use this function if there are no prio matches
 


    #----------------------Inserting the Graycode Matches Canon A to B into the TVG database -----------
    print("\n List TVGpairs \n -----  ")  
    
    canonCamA_B_TVGpairs_orig = db.execute("SELECT * from two_view_geometries where pair_id='"+str(ABpair_id)+"'")

    thepair_id, rows, cols, TVGdataBlob, config, F, E, H, extraBullshitA, extrabullshitB  = next(canonCamA_B_TVGpairs_orig)
    TVGarrayA_B_orig = np.frombuffer(TVGdataBlob, np.uint32).reshape(rows, cols)
    


    print("pair id for canon cameras images")
    print(thepair_id)
    print("original TVGpairs Cam A")
    print(TVGarrayA_B_orig)
    print(TVGarrayA_B_orig.shape)

    print("last index of orignal TVGpairs ")
    TVGpairsOrigIndexA_B =TVGarrayA_B_orig.shape[0] -1
    print(TVGpairsOrigIndexA_B)

    print("_------------------ADDING -----Augmented TVG Pairs ---Hopefully ")
    
    numofTVGpairsAddedA=totalNewKeypoints
    augmentedTVGpairsArrayAB = np.append(TVGarrayA_B_orig, newTVGpairs_A_B)
    print(numofTVGpairsAddedA)

    #reshape it back how the database likes it
    augmentedTVGpairsArrayAB =augmentedTVGpairsArrayAB.reshape(rows+numofTVGpairsAddedA,cols)

    
    print("TVGpairs data array augmented")
    print(augmentedTVGpairsArrayAB.shape)

    augmentedTVGpairsArrayAB = np.asarray(augmentedTVGpairsArrayAB, np.uint32)
    print(augmentedTVGpairsArrayAB.shape)
    print(augmentedTVGpairsArrayAB)

    #put those keypoints into that database!
    data_tuple = (ABpair_id, augmentedTVGpairsArrayAB.shape[0], augmentedTVGpairsArrayAB.shape[1],array_to_blob(augmentedTVGpairsArrayAB))
    
    #put the full list of matches into the database
    #canonCamA_B_Matches = db.execute("INSERT or REPLACE INTO matches (pair_id, rows, cols, data) VALUES (?, ?, ?, ?)",data_tuple)
    #canonCamA_B_TVGpairs_orig = db.execute("UPDATE two_view_geometries SET config=5 where pair_id=2147483701 ") #testing manually setting the config of the two view geoms


    #run some tests to make sure our indicies are correct.
    # so if everything is added correctly at the FIRST index of the augmented points we should have 
    # Cam A  879 , 33
    # 
    #
    # 
    #   
    #print("first graycode match")
    #print(augmentedTVGpairsArrayA[TVGpairsOrigIndexA_B+1])
    print("first graycode matches Keypoints")
    print(augmentedKeypointArrayA[ augmentedTVGpairsArrayAB[TVGpairsOrigIndexA_B+1][0]])
    print(augmentedKeypointArrayB[ augmentedTVGpairsArrayAB[TVGpairsOrigIndexA_B+1][1]])
   

    #add to two view geometries
    db.update_two_view_geometries(augmentedTVGpairsArrayAB, thepair_id)
    

    '''
 
    matches_CA_B = np.array([arrA,arrB])

    print("matches_CA_B:")
    print(matches_CA_B.shape)
    print(matches_CA_B)
    matches_CA_B =matches_CA_B.T #have to transpose it
    print(matches_CA_B.shape)
    print(matches_CA_B)

    db.update_two_view_geometries(camA_image_id,camB_image_id,matches_CA_B) 
    db.update_matches(camA_image_id,camB_image_id,matches_CA_B)    
    db.update_two_view_geometry(camA_image_id,camB_image_id,matches_CA_B) 
    '''

    """
    npAtoB = int_dfAtoB_noDups.to_numpy()
    #npAtoB = int_dfAtoB.to_numpy()
    print(npAtoB)
    idxs_A_B = np.delete(npAtoB,0,1) #kill first COLUMN
    idxs_A_B = np.delete(idxs_A_B,0,1) #kill the new first row COLUMN
    print("these are the indices that correspond to matches from Graycode File from CamA to CamB")
    print(idxs_A_B)
    print(idxs_A_B.shape)


    #Matches
    Matches_A_B = idxs_A_B 

    Matches_A_B = Matches_A_B +[canCamAKeypointsOrigIndex+1,canCamBKeypointsOrigIndex+1] #ADD THE KEYPOINT OFFSET TO EACH. A and then B #ALERT just added a plus 1!

    #TVGPairs
    TVGpairs_A_B = idxs_A_B 

    TVGpairs_A_B = TVGpairs_A_B +[canCamAKeypointsOrigIndex+1,canCamBKeypointsOrigIndex+1] #ADD THE KEYPOINT OFFSET TO EACH. A and then B #ALERT just added a plus 1!
    
    print("matches with correct offset for the keypoints")
    print(Matches_A_B)
    print(Matches_A_B.shape)

    print("TVGpairs with correct offset for the keypoints")
    print(TVGpairs_A_B)

    print(TVGpairs_A_B.shape)
    """



    #~~~~~ Commit the data to the file.
    
    
    db.commit()
    print("saved db")

    # Read and check cameras.
 
    rowsA = db.execute("SELECT * FROM cameras")
    camera_id, model, width, height, params, prior = next(rowsA)
    params = blob_to_array(params, np.float64)
    #print(params)

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
    print("~~~~~~~~finished graycode Goochin!~~~~~~")

    db.close()

    #if os.path.exists(args.database_path):
     #   os.remove(args.database_path)


if __name__ == "__main__":
    example_usage()
