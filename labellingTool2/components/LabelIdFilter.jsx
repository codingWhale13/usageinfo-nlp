import {
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  Switch,
  Button,
} from "@chakra-ui/react";
export default function LabelIdFilter({
  allLabelIds,
  selectedLabelIds,
  setSelectedLabelIds,
}) {
  return (
    <Menu closeOnSelect={false}>
      <MenuButton size="lg" as={Button}>
        Filter label ids
      </MenuButton>
      <MenuList>
        {allLabelIds.map((labelId) => {
          const is_selected = selectedLabelIds.includes(labelId);
          return (
            <MenuItem>
              <Switch
                isChecked={selectedLabelIds.includes(labelId)}
                onChange={() => {
                  if (is_selected) {
                    setSelectedLabelIds(
                      selectedLabelIds.filter((id) => id !== labelId)
                    );
                  } else {
                    setSelectedLabelIds([labelId, ...selectedLabelIds]);
                  }
                }}
              >
                {labelId}
              </Switch>
            </MenuItem>
          );
        })}
      </MenuList>
    </Menu>
  );
}
