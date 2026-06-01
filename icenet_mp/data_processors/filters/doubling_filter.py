from collections.abc import Iterator

import earthkit.data as ekd
from anemoi.transform.filters.fields.matching import MatchingFieldsFilter, MatchingSpec


class DoublingFilter(MatchingFieldsFilter):
    MATCHING: MatchingSpec = MatchingSpec(select="param", forward=("input_field",))

    def __init__(
        self,
        *,
        input_field: str = "msl",
        output_field: str = "doubled_msl",
    ) -> None:
        """An example filter that takes two arguments."""
        self.input_field = input_field
        self.output_field = output_field
        super().__init__(input_field=input_field)

    def forward_transform(self, input_field: ekd.Field) -> Iterator[ekd.Field]:
        """A forward transform that doubles the input field as a new field."""
        yield input_field
        yield self.new_field_from_numpy(
            input_field.to_numpy() * 2,
            template=input_field,
            param=self.output_field,
        )
