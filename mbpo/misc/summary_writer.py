import os
import six
import time
import torch

from tensorboard.compat import tf
from tensorboard.compat.proto.event_pb2 import SessionLog
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto import event_pb2
from tensorboard.plugins.projector.projector_config_pb2 import ProjectorConfig
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from torch.utils.tensorboard._convert_np import make_np
from torch.utils.tensorboard._embedding import (
    make_mat, make_sprite, make_tsv, write_pbtxt, get_embedding_info,
)
from torch.utils.tensorboard._onnx_graph import load_onnx_graph
from torch.utils.tensorboard._pytorch_graph import graph
from torch.utils.tensorboard._utils import figure_to_image
from torch.utils.tensorboard.summary import (
    scalar, histogram, histogram_raw, image, audio, text,
    pr_curve, pr_curve_raw, video, custom_scalars, image_boxes, mesh, hparams
)


class FileWriter(object):
    """Writes protocol buffers to event files to be consumed by TensorBoard.

    The `FileWriter` class provides a mechanism to create an event file in a
    given directory and add summaries and events to it. The class updates the
    file contents asynchronously. This allows a training program to call methods
    to add data to the file directly from the training loop, without slowing down
    training.
    """

    def __init__(self, log_dir, max_queue=10, flush_secs=120, filename_suffix=''):
        """Creates a `FileWriter` and an event file.
        On construction the writer creates a new event file in `log_dir`.
        The other arguments to the constructor control the asynchronous writes to
        the event file.

        Args:
          log_dir: A string. Directory where event file will be written.
          max_queue: Integer. Size of the queue for pending events and
            summaries before one of the 'add' calls forces a flush to disk.
            Default is ten items.
          flush_secs: Number. How often, in seconds, to flush the
            pending events and summaries to disk. Default is every two minutes.
          filename_suffix: A string. Suffix added to all event filenames
            in the log_dir directory. More details on filename construction in
            tensorboard.summary.writer.event_file_writer.EventFileWriter.
        """
        # Sometimes PosixPath is passed in and we need to coerce it to
        # a string in all cases
        # TODO: See if we can remove this in the future if we are
        # actually the ones passing in a PosixPath
        log_dir = str(log_dir)
        self.event_writer = EventFileWriter(
            log_dir, max_queue, flush_secs, filename_suffix)

    def get_logdir(self):
        """Returns the directory where event file will be written."""
        return self.event_writer.get_logdir()

    def add_event(self, event, step=None, walltime=None):
        """Adds an event to the event file.
        Args:
          event: An `Event` protocol buffer.
          step: Number. Optional global step value for training process
            to record with the event.
          walltime: float. Optional walltime to override the default (current)
            walltime (from time.time()) seconds after epoch
        """
        event.wall_time = time.time() if walltime is None else walltime
        if step is not None:
            # Make sure step is converted from numpy or other formats
            # since protobuf might not convert depending on version
            event.step = int(step)
        self.event_writer.add_event(event)

    def add_summary(self, summary, global_step=None, walltime=None):
        """Adds a `Summary` protocol buffer to the event file.
        This method wraps the provided summary in an `Event` protocol buffer
        and adds it to the event file.

        Args:
          summary: A `Summary` protocol buffer.
          global_step: Number. Optional global step value for training process
            to record with the summary.
          walltime: float. Optional walltime to override the default (current)
            walltime (from time.time()) seconds after epoch
        """
        event = event_pb2.Event(summary=summary)
        self.add_event(event, global_step, walltime)

    def add_graph(self, graph_profile, walltime=None):
        """Adds a `Graph` and step stats protocol buffer to the event file.

        Args:
          graph_profile: A `Graph` and step stats protocol buffer.
          walltime: float. Optional walltime to override the default (current)
            walltime (from time.time()) seconds after epoch
        """
        graph = graph_profile[0]
        stepstats = graph_profile[1]
        event = event_pb2.Event(graph_def=graph.SerializeToString())
        self.add_event(event, None, walltime)

        trm = event_pb2.TaggedRunMetadata(
            tag='step1', run_metadata=stepstats.SerializeToString())
        event = event_pb2.Event(tagged_run_metadata=trm)
        self.add_event(event, None, walltime)

    def add_onnx_graph(self, graph, walltime=None):
        """Adds a `Graph` protocol buffer to the event file.

        Args:
          graph: A `Graph` protocol buffer.
          walltime: float. Optional walltime to override the default (current)
            _get_file_writerfrom time.time())
        """
        event = event_pb2.Event(graph_def=graph.SerializeToString())
        self.add_event(event, None, walltime)

    def flush(self):
        """Flushes the event file to disk.
        Call this method to make sure that all pending events have been written to
        disk.
        """
        self.event_writer.flush()

    def close(self):
        """Flushes the event file to disk and close the file.
        Call this method when you do not need the summary writer anymore.
        """
        self.event_writer.close()

    def reopen(self):
        """Reopens the EventFileWriter.
        Can be called after `close()` to add more events in the same directory.
        The events will go into a new events file.
        Does nothing if the EventFileWriter was not closed.
        """
        self.event_writer.reopen()


class SummaryWriter(object):
    """Writes entries directly to event files in the log_dir to be
    consumed by TensorBoard.

    The `SummaryWriter` class provides a high-level API to create an event file
    in a given directory and add summaries and events to it. The class updates the
    file contents asynchronously. This allows a training program to call methods
    to add data to the file directly from the training loop, without slowing down
    training.
    """

    def __init__(self, log_dir=None, comment='', purge_step=None, max_queue=10,
                 flush_secs=120, filename_suffix=''):
        """Creates a `SummaryWriter` that will write out events and summaries
        to the event file.

        Args:
            log_dir (string): Save directory location. Default is
              runs/**CURRENT_DATETIME_HOSTNAME**, which changes after each run.
              Use hierarchical folder structure to compare
              between runs easily. e.g. pass in 'runs/exp1', 'runs/exp2', etc.
              for each new experiment to compare across them.
            comment (string): Comment log_dir suffix appended to the default
              ``log_dir``. If ``log_dir`` is assigned, this argument has no effect.
            purge_step (int):
              When logging crashes at step :math:`T+X` and restarts at step :math:`T`,
              any events whose global_step larger or equal to :math:`T` will be
              purged and hidden from TensorBoard.
              Note that crashed and resumed experiments should have the same ``log_dir``.
            max_queue (int): Size of the queue for pending events and
              summaries before one of the 'add' calls forces a flush to disk.
              Default is ten items.
            flush_secs (int): How often, in seconds, to flush the
              pending events and summaries to disk. Default is every two minutes.
            filename_suffix (string): Suffix added to all event filenames in
              the log_dir directory. More details on filename construction in
              tensorboard.summary.writer.event_file_writer.EventFileWriter.

        Examples::

            from torch.utils.tensorboard import SummaryWriter

            # create a summary writer with automatically generated folder name.
            writer = SummaryWriter()
            # folder location: runs/May04_22-14-54_s-MacBook-Pro.local/

            # create a summary writer using the specified folder name.
            writer = SummaryWriter("my_experiment")
            # folder location: my_experiment

            # create a summary writer with comment appended.
            writer = SummaryWriter(comment="LR_0.1_BATCH_16")
            # folder location: runs/May04_22-14-54_s-MacBook-Pro.localLR_0.1_BATCH_16/

        """
        torch._C._log_api_usage_once("tensorboard.create.summarywriter")
        if not log_dir:
            import socket
            from datetime import datetime
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            log_dir = os.path.join(
                'runs', current_time + '_' + socket.gethostname() + comment)
        self.log_dir = log_dir
        self.purge_step = purge_step
        self.max_queue = max_queue
        self.flush_secs = flush_secs
        self.filename_suffix = filename_suffix

        # Initialize the file writers, but they can be cleared out on close
        # and recreated later as needed.
        self.file_writer = self.all_writers = None
        self._get_file_writer()

        # Create default bins for histograms, see generate_testdata.py in tensorflow/tensorboard
        v = 1E-12
        buckets = []
        neg_buckets = []
        while v < 1E20:
            buckets.append(v)
            neg_buckets.append(-v)
            v *= 1.1
        self.default_bins = neg_buckets[::-1] + [0] + buckets

    def _check_caffe2_blob(self, item):
        """
        Caffe2 users have the option of passing a string representing the name of
        a blob in the workspace instead of passing the actual Tensor/array containing
        the numeric values. Thus, we need to check if we received a string as input
        instead of an actual Tensor/array, and if so, we need to fetch the Blob
        from the workspace corresponding to that name. Fetching can be done with the
        following:

        from caffe2.python import workspace (if not already imported)
        workspace.FetchBlob(blob_name)
        workspace.FetchBlobs([blob_name1, blob_name2, ...])
        """
        return isinstance(item, six.string_types)

    def _get_file_writer(self):
        """Returns the default FileWriter instance. Recreates it if closed."""
        if self.all_writers is None or self.file_writer is None:
            self.file_writer = FileWriter(self.log_dir, self.max_queue,
                                          self.flush_secs, self.filename_suffix)
            self.all_writers = {self.file_writer.get_logdir(): self.file_writer}
            if self.purge_step is not None:
                most_recent_step = self.purge_step
                self.file_writer.add_event(
                    Event(step=most_recent_step, file_version='brain.Event:2'))
                self.file_writer.add_event(
                    Event(step=most_recent_step, session_log=SessionLog(status=SessionLog.START)))
                self.purge_step = None
        return self.file_writer

    def get_logdir(self):
        """Returns the directory where event files will be written."""
        return self.log_dir

    def add_hparams(self, hparam_dict, metric_dict):
        """Add a set of hyperparameters to be compared in TensorBoard.

        Args:
            hparam_dict (dict): Each key-value pair in the dictionary is the
              name of the hyper parameter and it's corresponding value.
              The type of the value can be one of `bool`, `string`, `float`,
              `int`, or `None`.
            metric_dict (dict): Each key-value pair in the dictionary is the
              name of the metric and it's corresponding value. Note that the key used
              here should be unique in the tensorboard record. Otherwise the value
              you added by ``add_scalar`` will be displayed in hparam plugin. In most
              cases, this is unwanted.

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            with SummaryWriter() as w:
                for i in range(5):
                    w.add_hparams({'lr': 0.1*i, 'bsize': i},
                                  {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})

        Expected result:

        .. image:: _static/img/tensorboard/add_hparam.png
           :scale: 50 %

        """
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.add_scalar(k, v)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        torch._C._log_api_usage_once("tensorboard.logging.add_scalar")
        if self._check_caffe2_blob(scalar_value):
            scalar_value = workspace.FetchBlob(scalar_value)
        self._get_file_writer().add_summary(
            scalar(tag, scalar_value), global_step, walltime)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        """Adds many scalar data to summary.

        Args:
            main_tag (string): The parent name for the tags
            tag_scalar_dict (dict): Key-value pair storing the tag and corresponding values
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter()
            r = 5
            for i in range(100):
                writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
                                                'xcosx':i*np.cos(i/r),
                                                'tanx': np.tan(i/r)}, i)
            writer.close()
            # This call adds three values to the same scalar plot with the tag
            # 'run_14h' in TensorBoard's scalar section.

        Expected result:

        .. image:: _static/img/tensorboard/add_scalars.png
           :scale: 50 %

        """
        torch._C._log_api_usage_once("tensorboard.logging.add_scalars")
        walltime = time.time() if walltime is None else walltime
        fw_logdir = self._get_file_writer().get_logdir()
        for tag, scalar_value in tag_scalar_dict.items():
            fw_tag = fw_logdir + "/" + main_tag.replace("/", "_") + "_" + tag
            if fw_tag in self.all_writers.keys():
                fw = self.all_writers[fw_tag]
            else:
                fw = FileWriter(fw_tag, self.max_queue, self.flush_secs,
                                self.filename_suffix)
                self.all_writers[fw_tag] = fw
            if self._check_caffe2_blob(scalar_value):
                scalar_value = workspace.FetchBlob(scalar_value)
            fw.add_summary(scalar(main_tag, scalar_value),
                           global_step, walltime)

    def add_histogram(self, tag, values, global_step=None, bins='tensorflow', walltime=None, max_bins=None):
        torch._C._log_api_usage_once("tensorboard.logging.add_histogram")
        if self._check_caffe2_blob(values):
            values = workspace.FetchBlob(values)
        if isinstance(bins, six.string_types) and bins == 'tensorflow':
            bins = self.default_bins
        self._get_file_writer().add_summary(
            histogram(tag, values, bins, max_bins=max_bins), global_step, walltime)

    def add_histogram_raw(self, tag, min, max, num, sum, sum_squares,
                          bucket_limits, bucket_counts, global_step=None,
                          walltime=None):
        torch._C._log_api_usage_once("tensorboard.logging.add_histogram_raw")
        if len(bucket_limits) != len(bucket_counts):
            raise ValueError('len(bucket_limits) != len(bucket_counts), see the document.')
        self._get_file_writer().add_summary(
            histogram_raw(tag,
                          min,
                          max,
                          num,
                          sum,
                          sum_squares,
                          bucket_limits,
                          bucket_counts),
            global_step,
            walltime)

    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        torch._C._log_api_usage_once("tensorboard.logging.add_image")
        if self._check_caffe2_blob(img_tensor):
            img_tensor = workspace.FetchBlob(img_tensor)
        self._get_file_writer().add_summary(
            image(tag, img_tensor, dataformats=dataformats), global_step, walltime)

    def add_images(self, tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW'):
        torch._C._log_api_usage_once("tensorboard.logging.add_images")
        if self._check_caffe2_blob(img_tensor):
            img_tensor = workspace.FetchBlob(img_tensor)
        self._get_file_writer().add_summary(
            image(tag, img_tensor, dataformats=dataformats), global_step, walltime)

    def add_image_with_boxes(self, tag, img_tensor, box_tensor, global_step=None,
                             walltime=None, rescale=1, dataformats='CHW', labels=None):
        torch._C._log_api_usage_once("tensorboard.logging.add_image_with_boxes")
        if self._check_caffe2_blob(img_tensor):
            img_tensor = workspace.FetchBlob(img_tensor)
        if self._check_caffe2_blob(box_tensor):
            box_tensor = workspace.FetchBlob(box_tensor)
        if labels is not None:
            if isinstance(labels, str):
                labels = [labels]
            if len(labels) != box_tensor.shape[0]:
                labels = None
        self._get_file_writer().add_summary(image_boxes(
            tag, img_tensor, box_tensor, rescale=rescale, dataformats=dataformats, labels=labels), global_step, walltime)

    def add_figure(self, tag, figure, global_step=None, close=True, walltime=None):
        torch._C._log_api_usage_once("tensorboard.logging.add_figure")
        if isinstance(figure, list):
            self.add_image(tag, figure_to_image(figure, close), global_step, walltime, dataformats='NCHW')
        else:
            self.add_image(tag, figure_to_image(figure, close), global_step, walltime, dataformats='CHW')

    def add_video(self, tag, vid_tensor, global_step=None, fps=4, walltime=None):
        torch._C._log_api_usage_once("tensorboard.logging.add_video")
        self._get_file_writer().add_summary(
            video(tag, vid_tensor, fps), global_step, walltime)

    def add_audio(self, tag, snd_tensor, global_step=None, sample_rate=44100, walltime=None):
        torch._C._log_api_usage_once("tensorboard.logging.add_audio")
        if self._check_caffe2_blob(snd_tensor):
            snd_tensor = workspace.FetchBlob(snd_tensor)
        self._get_file_writer().add_summary(
            audio(tag, snd_tensor, sample_rate=sample_rate), global_step, walltime)

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        torch._C._log_api_usage_once("tensorboard.logging.add_text")
        self._get_file_writer().add_summary(
            text(tag, text_string), global_step, walltime)

    def add_onnx_graph(self, prototxt):
        torch._C._log_api_usage_once("tensorboard.logging.add_onnx_graph")
        self._get_file_writer().add_onnx_graph(load_onnx_graph(prototxt))

    def add_graph(self, model, input_to_model=None, verbose=False):
        # prohibit second call?
        # no, let tensorboard handle it and show its warning message.
        torch._C._log_api_usage_once("tensorboard.logging.add_graph")
        if hasattr(model, 'forward'):
            # A valid PyTorch model should have a 'forward' method
            self._get_file_writer().add_graph(graph(model, input_to_model, verbose))
        else:
            # Caffe2 models do not have the 'forward' method
            from caffe2.proto import caffe2_pb2
            from caffe2.python import core
            from torch.utils.tensorboard._caffe2_graph import (
                model_to_graph_def, nets_to_graph_def, protos_to_graph_def
            )
            if isinstance(model, list):
                if isinstance(model[0], core.Net):
                    current_graph = nets_to_graph_def(model)
                elif isinstance(model[0], caffe2_pb2.NetDef):
                    current_graph = protos_to_graph_def(model)
            else:
                # Handles cnn.CNNModelHelper, model_helper.ModelHelper
                current_graph = model_to_graph_def(model)
            event = event_pb2.Event(
                graph_def=current_graph.SerializeToString())
            self._get_file_writer().add_event(event)

    @staticmethod
    def _encode(rawstr):
        # I'd use urllib but, I'm unsure about the differences from python3 to python2, etc.
        retval = rawstr
        retval = retval.replace("%", "%%%02x" % (ord("%")))
        retval = retval.replace("/", "%%%02x" % (ord("/")))
        retval = retval.replace("\\", "%%%02x" % (ord("\\")))
        return retval

    def add_embedding(self, mat, metadata=None, label_img=None, global_step=None, tag='default', metadata_header=None):
        torch._C._log_api_usage_once("tensorboard.logging.add_embedding")
        mat = make_np(mat)
        if global_step is None:
            global_step = 0
            # clear pbtxt?

        # Maybe we should encode the tag so slashes don't trip us up?
        # I don't think this will mess us up, but better safe than sorry.
        subdir = "%s/%s" % (str(global_step).zfill(5), self._encode(tag))
        save_path = os.path.join(self._get_file_writer().get_logdir(), subdir)

        fs = tf.io.gfile.get_filesystem(save_path)
        if fs.exists(save_path):
            if fs.isdir(save_path):
                print(
                    'warning: Embedding dir exists, did you set global_step for add_embedding()?')
            else:
                raise Exception("Path: `%s` exists, but is a file. Cannot proceed." % save_path)
        else:
            fs.makedirs(save_path)

        if metadata is not None:
            assert mat.shape[0] == len(
                metadata), '#labels should equal with #data points'
            make_tsv(metadata, save_path, metadata_header=metadata_header)

        if label_img is not None:
            assert mat.shape[0] == label_img.shape[0], '#images should equal with #data points'
            make_sprite(label_img, save_path)

        assert mat.ndim == 2, 'mat should be 2D, where mat.size(0) is the number of data points'
        make_mat(mat, save_path)

        # Filesystem doesn't necessarily have append semantics, so we store an
        # internal buffer to append to and re-write whole file after each
        # embedding is added
        if not hasattr(self, "_projector_config"):
            self._projector_config = ProjectorConfig()
        embedding_info = get_embedding_info(
            metadata, label_img, fs, subdir, global_step, tag)
        self._projector_config.embeddings.extend([embedding_info])

        from google.protobuf import text_format
        config_pbtxt = text_format.MessageToString(self._projector_config)
        write_pbtxt(self._get_file_writer().get_logdir(), config_pbtxt)


    def add_pr_curve(self, tag, labels, predictions, global_step=None,
                     num_thresholds=127, weights=None, walltime=None):
        torch._C._log_api_usage_once("tensorboard.logging.add_pr_curve")
        labels, predictions = make_np(labels), make_np(predictions)
        self._get_file_writer().add_summary(
            pr_curve(tag, labels, predictions, num_thresholds, weights),
            global_step, walltime)

    def add_pr_curve_raw(self, tag, true_positive_counts,
                         false_positive_counts,
                         true_negative_counts,
                         false_negative_counts,
                         precision,
                         recall,
                         global_step=None,
                         num_thresholds=127,
                         weights=None,
                         walltime=None):
        torch._C._log_api_usage_once("tensorboard.logging.add_pr_curve_raw")
        self._get_file_writer().add_summary(
            pr_curve_raw(tag,
                         true_positive_counts,
                         false_positive_counts,
                         true_negative_counts,
                         false_negative_counts,
                         precision,
                         recall,
                         num_thresholds,
                         weights),
            global_step,
            walltime)

    def add_custom_scalars_multilinechart(self, tags, category='default', title='untitled'):
        torch._C._log_api_usage_once("tensorboard.logging.add_custom_scalars_multilinechart")
        layout = {category: {title: ['Multiline', tags]}}
        self._get_file_writer().add_summary(custom_scalars(layout))

    def add_custom_scalars_marginchart(self, tags, category='default', title='untitled'):
        torch._C._log_api_usage_once("tensorboard.logging.add_custom_scalars_marginchart")
        assert len(tags) == 3
        layout = {category: {title: ['Margin', tags]}}
        self._get_file_writer().add_summary(custom_scalars(layout))

    def add_custom_scalars(self, layout):
        torch._C._log_api_usage_once("tensorboard.logging.add_custom_scalars")
        self._get_file_writer().add_summary(custom_scalars(layout))

    def add_mesh(self, tag, vertices, colors=None, faces=None, config_dict=None, global_step=None, walltime=None):
        torch._C._log_api_usage_once("tensorboard.logging.add_mesh")
        self._get_file_writer().add_summary(mesh(tag, vertices, colors, faces, config_dict), global_step, walltime)

    def flush(self):
        if self.all_writers is None:
            return
        for writer in self.all_writers.values():
            writer.flush()

    def close(self):
        if self.all_writers is None:
            return  # ignore double close
        for writer in self.all_writers.values():
            writer.flush()
            writer.close()
        self.file_writer = self.all_writers = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
