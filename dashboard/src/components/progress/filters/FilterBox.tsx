import { Box, Typography, Chip, Button } from '@mui/material';
import { FilterOption } from '@/services/types';

interface FilterBoxProps {
    filters: FilterOption[];
    onClear: () => void;
    onApply: () => void;
    onFilterRemove: (type: string, value: string) => void;
    children: React.ReactNode;
}

export function FilterBox({
    filters,
    onClear,
    onApply,
    onFilterRemove,
    children
}: FilterBoxProps) {
    return (
        <Box sx={{
            mb: 3,
            bgcolor: '#f8f9fa',
            p: 2.5,
            borderRadius: 1,
            border: '1px solid #e0e0e0',
            height: 132,
            position: 'relative',
            boxSizing: 'border-box',
            overflow: 'hidden'
        }}>
            <Box sx={{
                display: 'flex',
                flexWrap: 'wrap',
                alignItems: 'center',
                gap: 1,
                height: 32,
                overflow: 'hidden',
                position: 'absolute',
                top: 20,
                left: 20,
                right: 20
            }}>
                <Typography variant="body2" sx={{ color: '#5f6368', mr: 1, fontSize: '14px', fontWeight: 500 }}>
                    Filters:
                </Typography>

                {filters.length > 0 ? (
                    filters.map((filter, index) => (
                        <Chip
                            key={`${filter.type}-${filter.value}-${index}`}
                            label={`${filter.type}:${filter.value}`}
                            onDelete={() => onFilterRemove(filter.type, filter.value)}
                            size="small"
                            sx={{
                                bgcolor: '#e8f0fe',
                                color: '#1a73e8',
                                height: 24,
                                '& .MuiChip-deleteIcon': {
                                    color: '#1a73e8',
                                }
                            }}
                        />
                    ))
                ) : (
                    <Typography variant="body2" color="text.secondary">
                        No filters applied
                    </Typography>
                )}
            </Box>

            <Box sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                position: 'absolute',
                bottom: 20,
                left: 20,
                right: 20
            }}>
                <Box sx={{ display: 'flex', gap: 1 }}>
                    {children}
                </Box>

                <Box sx={{ display: 'flex', gap: 1 }}>
                    <Button
                        variant="outlined"
                        size="small"
                        onClick={onClear}
                        sx={{ height: 42 }}
                    >
                        Clear
                    </Button>
                    <Button
                        variant="contained"
                        size="small"
                        onClick={onApply}
                        sx={{
                            height: 42,
                            bgcolor: '#1a73e8',
                            '&:hover': {
                                bgcolor: '#1765c6',
                            }
                        }}
                    >
                        Apply
                    </Button>
                </Box>
            </Box>
        </Box>
    );
} 