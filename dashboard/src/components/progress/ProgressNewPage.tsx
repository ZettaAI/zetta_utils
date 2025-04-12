'use client';

import { useState, useEffect } from 'react';
import { useRouter, useSearchParams, usePathname } from 'next/navigation';
import { Box, Paper, Typography, Tabs, Tab, Button, CircularProgress } from '@mui/material';
import { FilterBox } from './filters/FilterBox';
import { FilterDropdown } from './filters/FilterDropdown';
import { FilterOption, SubtaskTypeInfo } from '@/services/types';
import firebaseService from '@/services/FirebaseService';

export default function ProgressNewPage() {
    const router = useRouter();
    const pathname = usePathname();
    const searchParams = useSearchParams();
    const urlTab = searchParams.get('tab') || 'tasks';
    const projectId = searchParams.get('project') || '';

    const [tabValue, setTabValue] = useState(urlTab === 'subtasks' ? 1 : 0);
    const [subtaskTypeInfos, setSubtaskTypeInfos] = useState<SubtaskTypeInfo[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // States for selected filters (pending application)
    const [selectedSubtaskTypes, setSelectedSubtaskTypes] = useState<string[]>([]);
    const [selectedBatches, setSelectedBatches] = useState<string[]>([]);
    const [selectedTaskBatches, setSelectedTaskBatches] = useState<string[]>([]);

    // States for applied filters (used in API calls)
    const [appliedSubtaskTypes, setAppliedSubtaskTypes] = useState<string[]>([]);
    const [appliedBatches, setAppliedBatches] = useState<string[]>([]);
    const [appliedTaskBatches, setAppliedTaskBatches] = useState<string[]>([]);

    // Fetch subtask types when project changes
    useEffect(() => {
        const fetchSubtaskTypes = async () => {
            if (!projectId) return;

            setLoading(true);
            setError(null);
            try {
                const typeInfos = await firebaseService.getSubtaskTypesWithCounts(projectId);
                setSubtaskTypeInfos(typeInfos);

                // Only set types if none are selected
                if (selectedSubtaskTypes.length === 0) {
                    const allTypes = typeInfos.map(info => info.type);
                    setSelectedSubtaskTypes(allTypes);
                }
            } catch (error) {
                console.error('Error fetching subtask types:', error);
                setError('Failed to load subtask types');
            } finally {
                setLoading(false);
            }
        };

        fetchSubtaskTypes();
    }, [projectId]);

    const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
        setTabValue(newValue);
        const params = new URLSearchParams(searchParams);
        params.set('tab', newValue === 0 ? 'tasks' : 'subtasks');
        window.history.replaceState(null, '', `${pathname}?${params.toString()}`);
    };

    const handleSubtaskTypeChange = (event: any) => {
        const value = event.target.value;
        const types = typeof value === 'string' ? value.split(',') : value;
        setSelectedSubtaskTypes(types);
    };

    const handleBatchChange = (event: any) => {
        const value = event.target.value;
        const batches = typeof value === 'string' ? value.split(',') : value;
        setSelectedBatches(batches);
    };

    const handleTaskBatchChange = (event: any) => {
        const value = event.target.value;
        const batches = typeof value === 'string' ? value.split(',') : value;
        setSelectedTaskBatches(batches);
    };

    const applyFilters = () => {
        setAppliedSubtaskTypes(selectedSubtaskTypes);
        setAppliedBatches(selectedBatches);
    };

    const applyTaskFilters = () => {
        setAppliedTaskBatches(selectedTaskBatches);
    };

    // Convert selectedTaskBatches to filter format for the FilterBox component
    const taskFilters = selectedTaskBatches.map(batch => ({
        type: 'batch',
        value: batch
    }));

    // Convert selectedSubtaskTypes and selectedBatches to filter format
    const subtaskFilters = [
        ...selectedSubtaskTypes.map(type => ({ type: 'type', value: type })),
        ...selectedBatches.map(batch => ({ type: 'batch', value: batch }))
    ];

    // Filter handlers for removing chips
    const handleFilterRemove = (type: string, value: string) => {
        if (type === 'batch' && tabValue === 0) {
            setSelectedTaskBatches(prev => prev.filter(b => b !== value));
        } else if (tabValue === 1) {
            if (type === 'type') {
                setSelectedSubtaskTypes(prev => prev.filter(t => t !== value));
            } else if (type === 'batch') {
                setSelectedBatches(prev => prev.filter(b => b !== value));
            }
        }
    };

    // Batch options for both tabs
    const batchOptions = [
        { value: 'batch_1', label: 'batch_1' },
        { value: 'batch_2', label: 'batch_2' }
    ];

    if (!projectId) {
        return (
            <Box sx={{ p: 3 }}>
                <Paper sx={{ p: 3 }}>
                    <Typography>Please select a project to view progress.</Typography>
                </Paper>
            </Box>
        );
    }

    if (loading) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                <CircularProgress />
            </Box>
        );
    }

    if (error) {
        return (
            <Box sx={{ p: 3 }}>
                <Paper sx={{ p: 3 }}>
                    <Typography color="error">{error}</Typography>
                </Paper>
            </Box>
        );
    }

    return (
        <Box sx={{ p: 3 }}>
            <Paper sx={{ p: 3 }}>
                <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
                    <Tabs
                        value={tabValue}
                        onChange={handleTabChange}
                        aria-label="progress tabs"
                    >
                        <Tab label="Tasks" />
                        <Tab label="Subtasks" />
                    </Tabs>
                </Box>

                {tabValue === 0 && (
                    <>
                        <FilterBox
                            filters={taskFilters}
                            onClear={() => setSelectedTaskBatches([])}
                            onApply={applyTaskFilters}
                            onFilterRemove={handleFilterRemove}
                        >
                            <FilterDropdown
                                id="add-task-batch-filter"
                                label="Add batch filter"
                                values={batchOptions}
                                selected={selectedTaskBatches}
                                onChange={handleTaskBatchChange}
                            />
                        </FilterBox>
                        <Typography>Tasks view coming soon...</Typography>
                    </>
                )}
                {tabValue === 1 && (
                    <>
                        <FilterBox
                            filters={subtaskFilters}
                            onClear={() => {
                                setSelectedSubtaskTypes([]);
                                setSelectedBatches([]);
                            }}
                            onApply={applyFilters}
                            onFilterRemove={handleFilterRemove}
                        >
                            <FilterDropdown
                                id="add-batch-filter"
                                label="Add batch filter"
                                values={batchOptions}
                                selected={selectedBatches}
                                onChange={handleBatchChange}
                            />
                            <FilterDropdown
                                id="add-type-filter"
                                label="Add type filter"
                                values={subtaskTypeInfos.map(ti => ({ value: ti.type, label: ti.type }))}
                                selected={selectedSubtaskTypes}
                                onChange={handleSubtaskTypeChange}
                            />
                        </FilterBox>
                        <Typography>Subtasks view coming soon...</Typography>
                    </>
                )}
            </Paper>
        </Box>
    );
} 