import { useState, useEffect } from 'react';
import { Box, Paper, Tabs, Tab, Button, Typography, Chip, Checkbox, ListItemText, FormControl, Select, MenuItem, Alert } from '@mui/material';
import { getSubtaskTypesWithCounts, getTaskStatusCounts, getSubtaskStatusCounts } from '@/services/FirebaseService';
import { useRouter, useSearchParams } from 'next/navigation';

type SubtaskTypeInfo = {
    type: string;
    count: number;
};

type StatusCount = {
    status: string;
    count: number;
};

type FilterOption = {
    type: string;
    value: string;
};

function FilterBox({
    filters,
    onClear,
    onApply,
    onFilterRemove,
    children
}: {
    filters: { type: string, value: string }[],
    onClear: () => void,
    onApply: () => void,
    onFilterRemove: (type: string, value: string) => void,
    children: React.ReactNode
}) {
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

function FilterDropdown({
    id,
    label,
    values,
    selected,
    onChange
}: {
    id: string,
    label: string,
    values: { value: string, label: string }[],
    selected: string[],
    onChange: (event: any) => void
}) {
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

export default function ProgressNewPage() {
    const searchParams = useSearchParams();
    const projectId = searchParams.get('project') || '';

    const [tabValue, setTabValue] = useState(0);
    const [taskCounts, setTaskCounts] = useState<StatusCount[]>([]);
    const [subtaskCounts, setSubtaskCounts] = useState<StatusCount[]>([]);
    const [subtaskTypeInfos, setSubtaskTypeInfos] = useState<SubtaskTypeInfo[]>([]);

    const [selectedSubtaskTypes, setSelectedSubtaskTypes] = useState<FilterOption[]>([]);
    const [selectedBatches, setSelectedBatches] = useState<FilterOption[]>([]);
    const [selectedTaskBatches, setSelectedTaskBatches] = useState<FilterOption[]>([]);

    const [appliedSubtaskTypes, setAppliedSubtaskTypes] = useState<FilterOption[]>([]);
    const [appliedBatches, setAppliedBatches] = useState<FilterOption[]>([]);
    const [appliedTaskBatches, setAppliedTaskBatches] = useState<FilterOption[]>([]);

    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        let isMounted = true;

        const fetchSubtaskTypes = async () => {
            if (!projectId || !isMounted) return;

            try {
                const typeInfos = await getSubtaskTypesWithCounts(projectId);
                if (isMounted) {
                    setSubtaskTypeInfos(typeInfos);

                    if (typeInfos.length > 0 && selectedSubtaskTypes.length === 0) {
                        const allTypes = typeInfos.map(info => ({ type: info.type, value: info.type }));
                        setSelectedSubtaskTypes(allTypes);
                        setAppliedSubtaskTypes(allTypes);
                    }
                }
            } catch (error) {
                console.error('Error fetching subtask types:', error);
                if (isMounted) setError('Failed to load subtask types.');
            }
        };

        fetchSubtaskTypes();

        return () => {
            isMounted = false;
        };
    }, [projectId]);

    const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
        setTabValue(newValue);
    };

    const handleSubtaskTypeChange = (value: string | string[]) => {
        const types = typeof value === 'string' ? value.split(',') : value;
        setSelectedSubtaskTypes(types.map(type => ({ type, value: type })));
    };

    const handleBatchChange = (value: string | string[]) => {
        const batches = typeof value === 'string' ? value.split(',') : value;
        setSelectedBatches(batches.map(batch => ({ type: batch, value: batch })));
    };

    const handleTaskBatchChange = (value: string | string[]) => {
        const batches = typeof value === 'string' ? value.split(',') : value;
        setSelectedTaskBatches(batches.map(batch => ({ type: batch, value: batch })));
    };

    if (error) {
        return <Alert severity="error">{error}</Alert>;
    }

    return (
        <Box sx={{ width: '100%' }}>
            <Paper sx={{ width: '100%', mb: 2 }}>
                <Tabs
                    value={tabValue}
                    onChange={handleTabChange}
                    indicatorColor="primary"
                    textColor="primary"
                    sx={{ borderBottom: 1, borderColor: 'divider' }}
                >
                    <Tab label="Tasks" />
                    <Tab label="Subtasks" />
                </Tabs>
            </Paper>
            {tabValue === 0 && (
                <FilterBox
                    filters={selectedTaskBatches.map(batch => ({ type: batch.type, value: batch.value }))}
                    onClear={() => setSelectedTaskBatches([])}
                    onApply={() => { }}
                    onFilterRemove={(type, value) => setSelectedTaskBatches(selectedTaskBatches.filter(f => f.value !== value))}
                >
                    <FilterDropdown
                        id="task-batch-filter"
                        label="Batch Filter"
                        values={selectedTaskBatches.map(batch => ({ value: batch.value, label: batch.value }))}
                        selected={selectedTaskBatches.map(batch => batch.value)}
                        onChange={handleTaskBatchChange}
                    />
                </FilterBox>
            )}
            {tabValue === 1 && (
                <FilterBox
                    filters={selectedSubtaskTypes.map(type => ({ type: type.type, value: type.value }))}
                    onClear={() => setSelectedSubtaskTypes([])}
                    onApply={() => { }}
                    onFilterRemove={(type, value) => setSelectedSubtaskTypes(selectedSubtaskTypes.filter(f => f.value !== value))}
                >
                    <FilterDropdown
                        id="subtask-batch-filter"
                        label="Batch Filter"
                        values={selectedBatches.map(batch => ({ value: batch.value, label: batch.value }))}
                        selected={selectedBatches.map(batch => batch.value)}
                        onChange={handleBatchChange}
                    />
                    <FilterDropdown
                        id="subtask-type-filter"
                        label="Type Filter"
                        values={subtaskTypeInfos.map(info => ({ value: info.type, label: info.type }))}
                        selected={selectedSubtaskTypes.map(type => type.value)}
                        onChange={handleSubtaskTypeChange}
                    />
                </FilterBox>
            )}
        </Box>
    );
} 