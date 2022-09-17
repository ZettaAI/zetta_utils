from typeguard import typechecked
import torch
import attrs

from zetta_utils import builder


@builder.register("ApplyPerChannelRange")
@typechecked
@attrs.mutable
class ApplyPerChannelRange(torch.nn.Module):
    apply_fns: torch.nn.ModuleList = attrs.field(converter=torch.nn.ModuleList)
    channel_ranges: list[list[int]]

    def __attr_pre_init__(self):
        super().__init__()

    def __attr_post_init__(self):
        assert len(self.apply_fns) == len(self.channel_ranges)
        for e in self.channel_ranges:
            assert len(e) == 2

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        result_pieces = []
        for channel_range, apply_fn in zip(self.channel_ranges, self.apply_fns):
            data_piece = data[:, channel_range[0] : channel_range[1]]
            result_piece = apply_fn(data_piece)
            result_pieces.append(result_piece)

        result = torch.cat(result_pieces, dim=1)

        return result
