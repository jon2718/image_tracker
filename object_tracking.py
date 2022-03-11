__copyright__ = """

    Copyright 2022 Jon Lederman

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""
__license__ = "Apache 2.0"

from sklearn.mixture import GaussianMixture as GMM
import networkx as nx
import igraph as ig
import maxflow as mf
import math

class EnergyGraph:
    '''Receives a 2D image of GMM data and an image in BGR format'''
    def __init__(self, image, gmmData = None, neighbors = 'fourNode'):
        
        rows = image.shape[0]
        columns = image.shape[1]
        self.sigma2 = self.totalIntensityDifference(image)**2

        self.graph = nx.DiGraph()

        for row in range(rows):
            for column in range(columns):
                node = (row, column)

                #North
                if row > 0:
                    north = (row - 1, column)
                    self.graph.add_edge(node, north)

                #East
                if column < columns - 1:
                    east = (row, column + 1)
                    self.graph.add_edge(node, north)

                #South
                if row < rows - 1:
                    south = (row + 1, column)
                    self.graph.add_edge(node, south)

                #West
                if column > 0:
                    west = (row, column - 1)
                    self.graph.add_edge(node, west)

                if neighbors == 'eightNode':
                    pass

                #Add source and sink node edges
                self.graph.add_edge('s', node)
                self.graph.add_edge(node, 't')

    def Vpq(self, image, row, column, d = 1, neighbors = 'fournode'):
        d = 1
        rows = image.shape[0]
        columns = image.shape[1]
        twoSigma2 = 2*self.sigma2
        ip = self.intensity(image[row][column])
        vpq = 0
        if row > 0:
            iq = self.intensity(image[row - 1][column])
            vpq += math.exp((ip-iq)**2/self.sigma2)

        if row < rows - 1:
            iq = self.intensity(image[row + 1][column])
            vpq += math.exp((ip-iq)**2/self.sigma2)

        if column > 0:
            iq = self.intensity(image[row][column - 1])
            vpq += math.exp((ip-iq)**2/self.sigma2)

        if column < columns - 1:
            iq = self.intensity(image[row][column + 1])
            vpq += math.exp((ip-iq)**2/self.sigma2)
        
        if neighbors == 'eightnode':
            pass
        
        return vpq
        
        

    def totalNeighborIntensityDifference(self, image, row, column, neighbors = 'fourNode'):
        rows = image.shape[0]
        columns = image.shape[1]
        
        centerIntensity = self.intensity(image[row][column])

        neighboringIntensity = 0

        if row > 0:
            neighboringIntensity += self.intensity(image[row - 1][column]) - centerIntensity
        
        if column > 0:
            neighboringIntensity += self.intensity(image[row][column - 1]) - centerIntensity

        if row < rows - 1:
            neighboringIntensity += self.intensity(image[row + 1][column]) - centerIntensity

        if row < columns - 1:
            neighboringIntensity += self.intensity(image[row][column + 1]) - centerIntensity

        if neighbors == 'eightNode':
            pass  #TBD

        return neighboringIntensity

    def totalIntensityDifference(self, image):
        '''Assume image is in RGB'''
        ti = 0
        rows = image.shape[0]
        columns = image.shape[1]

        for row in rows:
            for column in columns:
                ti += self.totalNeighborIntensityDifference(image, row, column)
        
        return ti

    def intensity(self, rgbVector):
        return (1/3) * rgbVector.sum()


class Tracker:
    def __init__(self):
        '''
        Grab the labels from the Cord platform. We are picking random data to test, 
        but we can try other data later.
        We initialize the following properties:
            -boundingBoxes (bounding boxes for frames)
            -numTotalBoxes
            -numInSampleBoxes 
            -inSampleImages (BGR format)
            -outSampleImages (BGR format)
            -h (image height)
            -w (image width)
        '''
        client = CordClient.initialise(
            '3e71f710-7144-4631-9e51-fe36f44d6ad1',  # Project ID
            '1GuWzebRZyLQhnV01z85YfXW0EXna-7Ou2YglFYdeYc'  # API key
        )
  
        # Get project info (labels, datasets)
        project = client.get_project()
        label_blurb=client.get_label_row('aeca1f8f-919f-4214-811c-d345a4f2bf15')
        sequence_to_key = dict()
        for key in label_blurb['data_units'].keys():
            sequence_to_key[label_blurb['data_units'][key]['data_sequence']] = key

        '''Here we load in the "in sample" bounding boxes. Bounding box structure is a dict with keys:
        x: normalized leftmost cooridnate of box
        y: normalized topmost coordinate of box
        w: normalized width of box
        h: normalized height of box
        Normalization implies you must scale x by the frame width and y by frame height
        '''
        self.boundingBoxes = []
        self.numInSampleBoxes = 3

        for i in range(self.numInSampleBoxes):
            key = sequence_to_key[str(i)]
            self.boundingBoxes.append(label_blurb['data_units'][key]['labels']['objects'][0]['boundingBox'])

        self.numTotalBoxes = 20
        #color_array = []
        self.inSampleImages = []
        self.outSampleImages = []

        for i in range(0, self.numTotalBoxes):
            key = sequence_to_key[str(i)]
            url = label_blurb['data_units'][key]['data_link']
            image_file = io.BytesIO(urlopen(url).read())
            im = Image.open(image_file)
            image = np.array(im)

            if i < self.numInSampleBoxes:
                self.inSampleImages.append(image)
            else:
                self.outSampleImages.append(image)

        (self.h, self.w)=image.shape[:2]

    def getGMMFromBB(self, frameNumber, numComponents, multivariateDimensions = 3, 
    portion = 'foreground', convertToHSV = True, covarianceType = 'tied'):
        '''Returns GMM data for an image as a 1-D vector 
           To display as an image map, calling functions must assemble into
           2-D array.
           For image tracking use a 5D multivariate Gaussian with 3 color values and 2 spatial values.
           Use this function to get GMM from an entire image by setting portion to foreground!
        '''
        frame = self.inSampleImages[frameNumber]
        if convertToHSV:
            frame = self.BGR2HSV(frame)

        if portion == 'foreground':
            img = self.extractForeground(frame, self.boundingBoxes[frameNumber])
            img = img.reshape((-1, 3)) #Reshape to 1-D vector fore foreground only
        elif portion == 'background':
            img = self.extractBackground(frame, self.boundingBoxes[frameNumber])
        else:
            return None

        gmmModel = GMM(n_components = numComponents, covariance_type = covarianceType).fit(img)
        gmmLabels = gmmModel.predict(img)

        return gmmLabels
    
    def createGNNMapFromImage(self, frameNumber, numComponents, multivariateDimensions = 3, 
    convertToHSV = True, covarianceType = 'tied'):
        '''Returns GMM map for foreground and background '''

        fgGNN = self.getGMMFromBB(frameNumber, numComponents, multivariateDimensions, 
        'foreground', convertToHSV, covarianceType)
        bgGNN = self.getGMMFromBB(frameNumber, numComponents, multivariateDimensions, 
        'background', convertToHSV, covarianceType)
       
        #print(fgGNN.shape, bgGNN.shape)
        return fgGNN, bgGNN


    def getBoundingBoxCoordinates(self, bb):
        x0=int(self.w*bb['x'])
        x1=int(self.w*(bb['x'] + bb['w']))
        y0=int(self.h*bb['y'])
        y1=int(self.h*(bb['y'] + bb['h']))
        return x0, x1, y0, y1

    def BGR2HSV(self, frame):
        '''
        Why is color segmentation easier on HSV?

        The reason we use HSV colorspace for color detection/thresholding 
        over RGB/BGR is that HSV is more robust towards external lighting changes. 
        This means that in cases of minor changes in external lighting 
        (such as pale shadows,etc.) Hue values vary relatively lesser than RGB values.

        For example, two shades of red colour might have similar Hue values, 
        but widely different RGB values. In real life scenarios such as object 
        tracking based on colour,we need to make sure that our program 
        runs well irrespective of environmental changes as much as possible. 
        So, we prefer HSV colour thresholding over RGB. :)

        The big reason is that it separates color information (chroma) 
        from intensity or lighting (luma). Because value is separated, you can 
        construct a histogram or thresholding rules using only saturation and hue. 
        This in theory will work regardless of lighting changes in the value channel.
        In practice it is just a nice improvement. Even by singling out only the hue 
        you still have a very meaningful representation of the base color that will 
        likely work much better than RGB. The end result is a more robust color 
        thresholding over simpler parameters.

        Hue is a continuous representation of color so that 0 and 360 are the same
        hue which gives you more flexibility with the buckets you use in a histogram.
        Geometrically you can picture the HSV color space as a cone or cylinder with
        H being the degree, saturation being the radius, and value being the height. 
        See the HSV wikipedia page.

        When reading a color image file, OpenCV imread() reads as a NumPy array
        ndarray of row (height) x column (width) x color (3) . 
        The order of color is BGR (blue, green, red).

        BGR and RGB are not color spaces, they are just conventions for the order 
        of the different color channels. cv2.cvtColor(img, cv2.COLOR_BGR2RGB) doesn't 
        do any computations (like a conversion to say HSV would), it just switches 
        around the order. Any ordering would be valid - in reality, the three values 
        (red, green and blue) are stacked to form one pixel. You can arrange them any
        way you like, as long as you tell the display what order you gave it.

        OpenCV imread, imwrite and imshow indeed all work with the BGR order, so there
        is no need to change the order when you read an image with cv2.imread and
        then want to show it with cv2.imshow.d

        While BGR is used consistently throughout OpenCV, most other image processing
        libraries use the RGB ordering. If you want to use matplotlib's imshow but
        read the image with OpenCV, you would need to convert from BGR to RGB.

        '''
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        return hsv

    def HSV2BGR(self, frame):
        bgr = cv2.cvtColor(frame,cv2.COLOR_HSV2BGR)
        return bgr


    def createGraphFromImage(self, gmmData, neighbors = '4-Neighbor'):
        '''
        Image is any 2-D array of data associated with pixels
        Default is 4-Neighbor whereby N, E, S, W are neighbors.
        Optional is 8-Neighbor adding NE, NW, SE, SW
        '''
        energyGraph = nx.DiGraph()

        for row in gmmData.shape[0]:
            for colunn in gmmData.shape[1]:
                node = EnergyNode(gmmData[row][column])
                energyGraph.add_node(node)


    def extractForeground(self, image, bb):
        '''Returns the extracted bounding box image from image as 2-D array.
        In OpenCV, pixels are accessed by their (x, y)-coordinates.
        The origin, (0, 0), is located at the top-left of the image. OpenCV images 
        are zero-indexed, where the x-values go left-to-right (column number) 
        and y-values go top-to-bottom (row number).
        OpenCV, like many other libraries, treat matrix access in row-major order. 
        That means every access is defined as (row, column).  The row is the "y"
        coordinate.  If (x, y) means xth column and yth row, this means, we access 
        pixels as (y, x)! 
        '''
        (h,w)=image.shape[:2]
        x0, x1, y0, y1 = self.getBoundingBoxCoordinates(bb)
        extract = image[y0:y1, x0: x1, :]
        return extract

    def extractBackground(self, image, bb):
        x0, x1, y0, y1 = self.getBoundingBoxCoordinates(bb)

        bool_arr = np.ones(image.shape[:2], dtype=bool)
        bool_arr[y0:y1, x0: x1] = False
        background = image[bool_arr, :]
        return background

    def reconstruct(self, shape, foreground, bb, background):
        '''Reconstruct an image from foreground and background data'''
        width = shape[1]
        height = shape[0]
        channels = shape[2]

        foreground1D = foreground.reshape(-1, 3)
        
        x0, x1, y0, y1 = self.getBoundingBoxCoordinates(bb)

        image = np.zeros((height, width, channels), dtype=int)
        foregroundPointer = 0 
        backgroundPointer = 0

        for row in range(height):
            for column in range(width):
                if row >= y0 and row < y1 and column >= x0 and column < x1:
                    image[row][column] = foreground1D[foregroundPointer]
                    foregroundPointer += 1
                else:
                    image[row][column] = background[backgroundPointer]
                    backgroundPointer += 1
        
        return image

    def constructGmmMap(self, frameNumber, numForegroundComponents = 4,
                        numBackgroundComponents = 4, multivariateDimensions = 3,
                        convertToHSV = True, covarianceType = 'tied'):
      width = self.inSampleImages[frameNumber].shape[1]
      height = self.inSampleImages[frameNumber].shape[0]

      '''Currently support ONLY same number of components in background and foreground'''
      fgGNN, bgGNN = t.createGNNMapFromImage(frameNumber, numForegroundComponents) 
      x0, x1, y0, y1 = self.getBoundingBoxCoordinates(self.boundingBoxes[frameNumber])

      map = np.zeros((height, width), dtype=float)
      foregroundPointer = 0 
      backgroundPointer = 0

      for row in range(height):
            for column in range(width):
                if row >= y0 and row < y1 and column >= x0 and column < x1:
                    map[row][column] = fgGNN[foregroundPointer]
                    foregroundPointer += 1
                else:
                  map[row][column] = bgGNN[backgroundPointer]
                  backgroundPointer += 1
        
      return map

