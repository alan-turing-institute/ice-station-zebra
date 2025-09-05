from collections.abc import Iterator

import earthkit.data as ekd  # type: ignore[import-untyped]
from anemoi.transform.filters.matching import MatchingFieldsFilter, matching


# @filter_registry.register("example_filter")
class ExampleFilter(MatchingFieldsFilter):
    @matching(
        select="param",
        forward=["input_field"],
    )
    def __init__(
        self,
        *,
        input_field: str = "msl",
    ) -> None:
        """Create an example filter."""
        self.input_field = input_field

    def forward_transform(self, input_field: ekd.Field) -> Iterator[ekd.Field]:
        """An example forward transform that doubles the input field."""
        yield input_field
        yield self.new_field_from_numpy(
            input_field.to_numpy() * 2, template=input_field, param="test_field"
        )

    def backward_transform(self, input_field: ekd.Field) -> Iterator[ekd.Field]:
        yield input_field
