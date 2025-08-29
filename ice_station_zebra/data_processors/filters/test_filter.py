import earthkit.data as ekd

from anemoi.transform.filters.matching import MatchingFieldsFilter
from anemoi.transform.filters import filter_registry
from anemoi.transform.fields import new_field_with_metadata
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filters.matching import matching
from collections.abc import Iterator


# @filter_registry.register("test_filter")
class testFilter(MatchingFieldsFilter):
    
    @matching(
        select="param",
        forward=["input_field"],
        # backward=("wind_speed", "wind_direction"),
    )
    def __init__(
        self,
        *,
        input_field: str = "msl",
    ) -> None:

        self.input_field = input_field

    def forward_transform(self, input_field: ekd.Field) -> Iterator[ekd.Field]:
        print("In test filter")
        # result = []
        print(f"Field: {input_field.metadata('param')}")
        print(f"Numpy: {input_field.to_numpy()}")
        print(f"Numpy shape: {input_field.to_numpy().shape}")
        #     if field.
        #     for _, renamer in self._rename.items():
        #         field = renamer.rename(field)
        #     result.append(field)

        # return new_fieldlist_from_list(result)
        yield input_field
        yield self.new_field_from_numpy(
            input_field.to_numpy() * 2, template = input_field, param="test_field"
        )
        
    def backward_transform(self, input_field: ekd.Field) -> Iterator[ekd.Field]:
        yield input_field