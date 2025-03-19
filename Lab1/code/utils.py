#coding=utf-8
import cv2
import numpy as np
import itertools
import matplotlib.pyplot as plt

def display_image(mat, axes=None, cmap=None, hide_axis=True):
    """
    Display a given matrix into Jupyter's notebook
    
    :param mat: Matrix to display
    :param axes: Subplot on which to display the image
    :param cmap: Color scheme to use
    :param hide_axis: If `True` axis ticks will be hidden
    :return: Matplotlib handle
    """
    img = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB) if mat.ndim == 3 else mat
    cmap= cmap if mat.ndim != 2 or cmap is not None else 'gray'
    if axes is None:
        if hide_axis:
            plt.xticks([])
            plt.yticks([])
        return plt.imshow(img, cmap=cmap)
    else:
        if hide_axis:
            axes.set_xticks([])
            axes.set_yticks([])
        return axes.imshow(img, cmap=cmap)
    
    
def plot_confusion_matrix(cm, 
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          axes=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    :param cm: Confusion matrix
    :param classes: Classes name
    :param normalize: Indicate if the confusion matrix need to be normalized
    :param title: Plot's title
    :param axes: Subplot on which to display the image
    :param cmap: Colormap to use
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    # Show cm
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    #fmt = '.2f' if normalize else 'd'
    #thresh = cm.max() / 2.
    #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #    plt.text(j, i, format(cm[i, j], fmt),
    #             horizontalalignment="center",
    #             color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
class ParticleFilterInterface:

    def __init__(self,
                 model,
                 search_space,
                 n_particles,
                 state_dims,
                 sigma_perturbation,
                 sigma_similarity,
                 alpha,
                 **kwargs):
        """
        Constrcutor
        :param model:         Template image to be tracked (numpy.ndarray)
        :param search_space:  Possible size of the search space (i.e. image size)
        :param n_particles:   Number of particle used to perform tracking
        :param state_dims:    State space dimensions (i.e. number of parameters
                              being tracked)
        :param sigma_perturbation: How much each particle will be perturbate (i.e. noise)
        :param alpha: Model adaptation, blending factor
        """
        # `Model` of the object being tracked (i.e. template)
        self.model = np.copy(model)
        # Range of search space (i.e. image dimensions)
        self.search_space = search_space
        # Number of particles used to estimate object position
        self.n_particles = n_particles
        # Number of state being estimated (will be equal to 2 => x, y)
        self.state_dims = state_dims
        # Magnitude of the perturbation added to the particles
        self.sigma_perturbation = sigma_perturbation
        # Similarity residual variance (i.e. eps = N(0, sigma_similarity ** 2.0))
        self.sigma_similarity = sigma_similarity
        # Blending coefficient for model adaptation
        self.alpha = alpha
        # Number of frames processed so far
        self.frame_counter = 0
        # 2D array particles stored as [[x0, y0], ..., [xN, yN]]
        self.particles = None
        # 1D array weights, probability of particles being at the real object
        # location
        self.weights = None
        # Current best estimation of the position, 1D array [x, y]
        self.state = None
        # Index of each particles
        self.indexes = np.arange(n_particles)
        # Toggle on/off sanity check, default is on
        self.verbose = kwargs.get('verbose', True)

    def current_state(self):
        """
        Select the current particles with the highest probability of being at the
        correct location of the object being tracked
        :return:  1D array, best state in form of [x, y] position
        """
        state_idx = np.random.choice(self.indexes, p=self.weights)
        return self.particles[state_idx, :]

    def perturbate(self):
        """ Perturbate particles by adding random normal noise """
        self.perturbate_impl()
        # Sanity check goes here
        if self.verbose:
          self._check_particle_dims()

    def perturbate_impl(self):
        """ Implementation of `Perturbate` function """
        raise NotImplementedError('Must be implemented by subclass')


    def reweight(self, frame):
        """
        Update particle's weight for the current frame. Check the similarity between
        every particles and the model of the tracked object.
        :param frame: New frame in the tracking sequence
        """
        self.reweight_impl(frame)
        if self.verbose:
            self._check_weights()

    def reweight_impl(self, frame):
        raise NotImplementedError('Must be implemented by sublass')

    def resample(self):
        """
        Draw a new set of particles using the probability distribution of the
        new weights.
        """
        self.resample_impl()
        if self.verbose:
            self._check_particle_dims(init=False)
            # Check particles is not outside image
            sh, sw = self.search_space[:2]
            msg = 'Particle larger than image width'
            assert np.any(self.particles[:, 0] < sw), msg
            msg = 'Particle larger than image height'
            assert np.any(self.particles[:, 1] < sh), msg

    def resample_impl(self):
        raise NotImplementedError('Must be implemented by sublclass')

    def update_model(self, frame):
        """
        Update tracking object model using current estimation and frame.
        :param frame: Current frame
        """
        shp = self.model.shape
        self.update_model_impl(frame)
        if self.verbose:
            msg = 'Update model must have same shape as before'
            assert self.model.shape == shp, msg

    def update_model_impl(self, frame):
        raise NotImplementedError('Must be implemented by sublclass')


    def draw_particles(self, image, color=(180, 255, 0)):
        """
        Draw current estimation of the tracked object by each individual particles
        :param image: Image to draw on
        :param color: Tuple, color to draw with
        :return: Updated image with particles draw on it
        """
        for p in self.particles:
            cv2.circle(image,
                       tuple(p.astype(int)),
                       radius=4,
                       color=color,
                       thickness=-1,
                       lineType=cv2.LINE_AA)
        return image

    def draw_window(self, image):
        """
        Draw current estimation of the tracked object using the best particles
        (i.e. The one with the highest probability)
        :param image: Image to draw on
        :return: Image with object position
        """
        best_idx = self.weights.argmax()
        best_state = self.state #self.particles[best_idx, :]
        pt1 = (best_state - np.asarray(self.model.shape[1::-1]) / 2).astype(np.int)
        #  pt1 = (self.state - np.array(self.model.shape[1::-1])/2).astype(np.int)
        pt2 = pt1 + np.asarray(self.model.shape[1::-1])
        cv2.rectangle(image,
                      tuple(pt1),
                      tuple(pt2),
                      color=(0, 255, 0), thickness=2,
                      lineType=cv2.LINE_AA)
        return image

    def draw_std(self, image):
        """
        Draw standard deviation between current state and all particles
        :param image: Canvas on which to draw
        :return:  Updated canvas
        """
        dist = np.linalg.norm(self.particles - self.state)
        weighted_sum = np.sum(dist * self.weights.reshape((-1, 1)))
        cv2.circle(image,
                   tuple(self.state.astype(np.int)),
                   int(weighted_sum),
                   (255, 255, 255),
                   1,
                   lineType=cv2.LINE_AA)
        return image

    def visualize_filter(self, image):
        """
        Visualize internal parts of the filter such as:
          - particles state
          - Current estimation of the object position (i.e. state)
          - Standard deviation between particles and estimation
          - Object's model
        :param image: Image to draw on
        :return:  Updated canvas
        """
        canvas = self.draw_particles(np.copy(image))
        canvas = self.draw_window(canvas)
        canvas = self.draw_std(canvas)
        # Add model in the top left corner
        canvas[:self.model.shape[0], :self.model.shape[1]] = self.model
        return canvas

    def _check_particle_dims(self, init=False):
        msg = 'particles dimensions must be (n_particle, state_dims)'
        assert self.particles.shape == (self.n_particles, self.state_dims), msg
        if init:
            p_max = self.particles.max(axis=0)
            msg = 'particle state 0 must be smaller than ' \
                  'search space: {}'.format(self.search_space[1])
            assert p_max[0] < self.search_space[1], msg
            msg = 'particle state 0 must be smaller than ' \
                  'search space: {}'.format(self.search_space[0])
            assert p_max[1] < self.search_space[0], msg

    def _check_weights(self):
          msg = 'weights dimensions must be equal to the number of particles, (n,)'
          assert self.weights.shape == (self.n_particles,), msg
          # Must be valid probability distribution
          assert np.abs(self.weights.sum() - 1.0) < 1e-6, 'Weight must sum to 1.0'
    

_tracker_ctor = {'mil': cv2.TrackerMIL_create,
                 'kcf': cv2.TrackerKCF_create,
                 'tld': cv2.TrackerTLD_create,
                 'medianflow': cv2.TrackerMedianFlow_create,
                 'mosse': cv2.TrackerMOSSE_create,
                 'goturn': cv2.TrackerGOTURN_create}
    
def create_face_tracker(name='KCF'):
    """
    Create an instance of a face tracker from a given `name`. The list of available tracker is :
    
    ['MIL', 'KCF', 'TLD', 'MedianFlow', 'Mosse', 'GoTurn']
    
    :param name: Name of the tracker instance to create
    :raise: ValueError exception if the name of the tracker do not match anything known.
    """
    n = name.lower()
    ctor = _tracker_ctor.get(n, None)
    if ctor is None:
        raise ValueError('Unknown type of tracker')
    return ctor()
        
    