from sklearn.mixture import GaussianMixture as GMM
import NetworkX as nx
import igraph as ig
import PyMaxflow as pmf

class EnergyGraph:
    def __init__(self, rows, columns, gmmData = None, neighbors = 'fourNode'):
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



class Tracker:
    def __init__(self):
        '''
        Grab the labels from the Cord platform. We are picking random data to test, 
        but we can try other data later.
        We initialize the following properties:
            -bounding_boxes (bounding boxes for frames)
            -num_total_boxes
            -num_in_sample_boxes 
            -in_sample_images (BGR format)
            -out_sample_images (BGR format)
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
        '''Returns GMM data for an image as a 1-D vector'''

        frame = self.in_sample_imges[frameNumber]
        if convertToHSV:
            frame = self.BGR2HSV(frame)

        if portion == 'foreground':
            img = self.extractForeground(frame, self.boundingBoxes[frameNumber])
        elif portion == 'background':
            img = self.extractForeground(frame, self.boundingBoxes[frameNumber])
        else:
            return None

        imgReshape = img.reshape((-1,3))
        gmmModel = GMM(n_components = numComponents, covariance_type = covarianceType).fit(imgReshape)
        gmmLabels = gmmModel.predict(imgReshape)
        
        #We are not returning GMM data as image currently
        originalShape = img.shape
        gmmDataAsImage = gmmLabels.reshape(originalShape[0], originalShape[1])
        #return gmmDataAsImage
        
        return gmmLabels
    
    def createGNNMapFromImage(self, frameNumber, numComponents, multivariateDimensions = 3, 
    convertToHSV = True, covarianceType = 'tied'):
        '''Returns GMM map from an image as a 2-D vector'''

        fgGNN = self.getGMMFromBB(frameNumber, numComponents, multivariateDimensions, 
        'foreground', convertToHSV, covarianceType)
        bgGNN = self.getGMMFromBB(frameNumber, numComponents, multivariateDimensions, 
        'background', convertToHSV, covarianceType)
        image =  self.reconstruct(self.in_sample_imges[frameNumber].shape, fgGNN, 
        self.boundingBoxes[frameNumber], bgGNN)

        return image




    
    def getBoundingBoxCoordinates(self, bb):
        x0=int(w*bb['x'])
        x1=int(w*(bb['x'] + bb['w']))
        y0=int(h*bb['y'])
        y1=int(h*(bb['y'] + bb['h']))
        return x0, x1, y0, y1

    def BGR2HSV(self, frame):
        '''
        Why is color segmentation easier on HSV?

        The reason we use HSV colorspace for color detection/thresholding 
        over RGB/BGR is that HSV is more robust towards external lighting changes. 
        This means that in cases of minor changes in external lighting (such as pale shadows,etc.) 
        Hue values vary relatively lesser than RGB values.

        For example, two shades of red colour might have similar Hue values, but widely different RGB values. 
        In real life scenarios such as object tracking based on colour,we need to make sure that our program 
        runs well irrespective of environmental changes as much as possible. So, we prefer HSV colour 
        thresholding over RGB. :)

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
        '''Returns the extracted bounding box image from image
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
        x0, x1, y0, y1 = getBoundingBoxCoordinates(bb)
        extract = image[y0:y1, x0: x1, :]
        return extract

    def extractBackground(self, image, bb):
        x0, x1, y0, y1 = getBoundingBoxCoordinates(bb)

        bool_arr = np.ones(image.shape[:2], dtype=bool)
        bool_arr[y0:y1, x0: x1] = False
        background = image[bool_arr, :]
        return background

    def reconstruct(self, shape, foreground, bb, background):
        '''Reconstruct an image from foreground and background data'''
        width = shape[1]
        height = shape[0]
        channels = shape[2]
        
        x0, x1, y0, y1 = getBoundingBoxCoordinates(bb)

        image = np.zeros((height, width, channels), dtype=int)
        foregroundPointer = 0 
        backgroundPointer = 0

        for row in range(height):
            for column in range(width):
            if row >= y0 and row < y1 and column >= x0 and column < x1:
                image[row][column] = foreground[foregroundPointer]
                foregroundPointer += 1
            else:
                image[row][column] = background[backgroundPointer]
                backgroundPointer += 1
        
        return image

