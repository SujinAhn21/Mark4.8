from vild_parser_common import BaseAudioParser


class AudioParser(BaseAudioParser):
    def __init__(self, config, segment_mode=False):
        super().__init__(config)
        self.segment_mode = segment_mode
