# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import os
LOG = True
try:
  import tensorflow as tf
except:
  LOG = False
import numpy as np
import scipy.misc 
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

def log_parameters(args):
    folder = args.save_path
    f = open(os.path.join(folder, 'parameters.log'), 'w+') 
    f.write('SourceDataset: %s\n' % args.sourceDataset)
    f.write('TargetDataset: %s\n' % args.targetDataset)
    f.write('PropsFiles: %s\n' % args.propsFile)
    f.write('LR: %.5f\n' % args.LR)
    f.write('DropLR: %d\n' % args.dropLR)
    f.write('normalized: %r\n' % (not args.unnormalized))
    f.write('distsRefiner: %s\n' % (args.distsRefiner if args.distsRefiner is not None else 'Identity'))
    f.write('logDir: %s\n' % args.logDir)
    f.write('expID: %s \n' % args.expID)
    f.write('debug: %s\n' % args.debug_folder)

class Logger(object):
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        if LOG:
          self.writer = tf.summary.FileWriter(log_dir)
          self.f = open(log_dir + '/log.txt', 'w')
        else:
          os.mkdir(log_dir)
          self.f = open(log_dir + '/log.txt', 'w')
    def write(self, txt):
        self.f.write(txt)
    
    def close(self):
        self.f.close()
    
    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        if LOG:
          summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
          self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)
        
    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
