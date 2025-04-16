import { FormControl, Select, MenuItem, Checkbox, ListItemText, Typography } from '@mui/material';

interface FilterDropdownProps {
    id: string;
    label: string;
    values: { value: string; label: string }[];
    selected: string[];
    onChange: (event: any) => void;
}

export function FilterDropdown({
    id,
    label,
    values,
    selected,
    onChange
}: FilterDropdownProps) {
    return (
        <FormControl size="small" sx={{ width: 160 }}>
            <Select
                id={id}
                multiple
                value={selected}
                onChange={onChange}
                displayEmpty
                sx={{
                    height: 42,
                    width: '100%',
                    '& .MuiSelect-select': {
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'flex-start',
                        paddingY: '10px',
                        paddingX: '12px'
                    }
                }}
                renderValue={() => (
                    <Typography
                        variant="body2"
                        sx={{
                            whiteSpace: 'nowrap',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            fontSize: '0.875rem',
                            lineHeight: '1.5'
                        }}
                    >
                        {label}
                    </Typography>
                )}
            >
                {values.map(item => (
                    <MenuItem key={item.value} value={item.value}>
                        <Checkbox checked={selected.includes(item.value)} />
                        <ListItemText primary={item.label} />
                    </MenuItem>
                ))}
            </Select>
        </FormControl>
    );
} 