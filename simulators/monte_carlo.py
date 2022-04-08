import numpy as np

from channel_codes import ClassicalErrorCorrection
from channels import ClassicalChannel


class MonteCarloSimulator:

    """Monte Carlo Simulator for frame error rate estimation."""
    def __init__(self, error_coder: ClassicalErrorCorrection, channel: ClassicalChannel):

        """Initialization.
        Args:
            error_coder: Error Correction Scheme.
            channel: Channel Model.
        """
        self.error_coder = error_coder
        self.channel = channel
        self.msg_frames = None
        self.channel_inputs = None
        self.decoded_frames = None
        self.channel_outputs = None
        self.num_frame_errors = None
        self.frame_error_rate = None

    def run(self, numtrials=1e7) -> float:

        """Run simulator.
        Args:
            numtrials: Number of Monte Carlo trials.
        Returns:
            Frame error rate for given error correction scheme under given channel.
        """
        self.msg_frames = np.random.randint(2, size=(numtrials, self.error_coder.msg_length))
        self.channel_inputs = self.error_coder.encode(self.msg_frames)
        self.channel_outputs = self.channel.transmit(self.channel_inputs)
        self.decoded_frames = self.error_coder.decode(self.channel_outputs)
        self.num_frame_errors = sum(np.any(self.msg_frames != self.decoded_frames, axis=1).astype(int))
        self.frame_error_rate = self.num_frame_errors / numtrials
        return self.frame_error_rate
