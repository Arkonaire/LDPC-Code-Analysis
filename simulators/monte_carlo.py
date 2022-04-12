import numpy as np

from decoders import LDPCDecoder
from encoders import LDPCEncoder
from channels import ClassicalChannel


class MonteCarloSimulator:

    """Monte Carlo Simulator for frame error rate estimation."""
    def __init__(self, encoder: LDPCEncoder, decoder: LDPCDecoder, channel: ClassicalChannel):

        """Initialization.
        Args:
            encoder: LDPC Encoder.
            decoder: LDPC Decoder.
            channel: Channel Model.
        """
        self.encoder = encoder
        self.decoder = decoder
        self.channel = channel
        self.msg_frames = None
        self.channel_inputs = None
        self.decoded_frames = None
        self.channel_outputs = None
        self.num_frame_errors = None
        self.frame_error_rate = None

    def run(self, numtrials=int(1e4)) -> float:

        """Run simulator.
        Args:
            numtrials: Number of Monte Carlo trials.
        Returns:
            Frame error rate for given error correction scheme under given channel.
        """
        self.msg_frames = np.random.randint(2, size=(numtrials, self.encoder.msg_length))
        self.channel_inputs = self.encoder.encode(self.msg_frames)
        self.channel_outputs = self.channel.transmit(self.channel_inputs)
        self.decoded_frames = self.decoder.decode(self.channel_outputs)
        self.num_frame_errors = sum(np.any(self.channel_inputs != self.decoded_frames, axis=1).astype(int))
        self.frame_error_rate = self.num_frame_errors / numtrials
        return self.frame_error_rate
