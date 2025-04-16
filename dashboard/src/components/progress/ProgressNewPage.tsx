'use client';

import { useState, useEffect } from 'react';
import { useRouter, useSearchParams, usePathname } from 'next/navigation';
import { Box, Paper, Typography, Tabs, Tab, Button, CircularProgress } from '@mui/material';
import { FilterBox } from './filters/FilterBox';
import { FilterDropdown } from './filters/FilterDropdown';
import { FilterOption, SubtaskTypeInfo, StatusCount } from '@/services/types';
import firebaseService from '@/services/FirebaseService';
import { ArcElement, Chart as ChartJS, Legend, Tooltip } from 'chart.js';
import { Pie } from 'react-chartjs-2';

ChartJS.register(ArcElement, Tooltip, Legend);

// Task status colors and labels
const taskStatusColors = {
    'pending_ingestion': '#FFA726', // Orange
    'ingested': '#42A5F5',         // Blue
    'fully_processed': '#66BB6A'    // Green
};

const subtaskStatusColors = {
    'pending_ingestion': '#FFA726',    // Orange
    'fully_processed': '#66BB6A'       // Green
};

const taskStatusLabels = {
    'pending_ingestion': 'Pending Ingestion',
    'ingested': 'Ingested',
    'fully_processed': 'Fully Processed'
};

const subtaskStatusLabels = {
    'pending_ingestion': 'Pending',
    'fully_processed': 'Completed'
};

// Task status types
export type TaskStatus = 'pending_ingestion' | 'ingested' | 'fully_processed';
export type SubtaskStatus = 'fully_processed' | 'pending_ingestion';

interface ChartData {
    labels: string[];
    datasets: {
        data: number[];
        backgroundColor: string[];
        borderColor: string[];
        borderWidth: number;
    }[];
}

interface GenericPieChartProps {
    data: ChartData;
    title?: string;
}

const typeColors = [
    '#42A5F5', // Blue
    '#66BB6A', // Green  
    '#FFA726', // Orange
    '#EF5350', // Red
    '#AB47BC', // Purple
    '#EC407A', // Pink
    '#7E57C2', // Deep Purple
    '#26A69A', // Teal
];

const GenericPieChart: React.FC<GenericPieChartProps> = ({ data, title }) => {
    const options = {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
            legend: {
                position: 'right' as const,
                align: 'center' as const,
            },
            tooltip: {
                callbacks: {
                    label: function (context: any) {
                        const label = context.label || '';
                        const value = context.raw || 0;
                        const total = context.dataset.data.reduce((a: number, b: number) => a + b, 0);
                        const percentage = total > 0 ? ((value / total) * 100).toFixed(1) + '%' : '0%';
                        return `${label}: ${value} (${percentage})`;
                    }
                }
            }
        },
    };

    return (
        <Box sx={{ height: 300, width: '100%', position: 'relative', display: 'flex', flexDirection: 'column' }}>
            {title && (
                <Typography variant="h6" gutterBottom align="center">
                    {title}
                </Typography>
            )}
            <Box sx={{ flex: 1, minHeight: 0 }}>
                <Pie data={data} options={options} />
            </Box>
        </Box>
    );
};

const NoDataDisplay: React.FC<{ type?: string }> = ({ type = 'data' }) => (
    <Box display="flex" justifyContent="center" alignItems="center" height="100%" minHeight={200}>
        <Typography variant="body1" color="textSecondary">
            No {type} available
        </Typography>
    </Box>
);

export default function ProgressNewPage() {
    const router = useRouter();
    const pathname = usePathname();
    const searchParams = useSearchParams();
    const urlTab = searchParams.get('tab') || 'tasks';
    const projectId = searchParams.get('project') || '';

    const [tabValue, setTabValue] = useState(urlTab === 'subtasks' ? 1 : 0);
    const [subtaskTypeInfos, setSubtaskTypeInfos] = useState<SubtaskTypeInfo[]>([]);
    const [subtaskTypesLoading, setSubtaskTypesLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // States for selected filters (pending application)
    const [selectedSubtaskTypes, setSelectedSubtaskTypes] = useState<string[]>([]);
    const [selectedBatches, setSelectedBatches] = useState<string[]>([]);
    const [selectedTaskBatches, setSelectedTaskBatches] = useState<string[]>([]);

    // States for applied filters (used in API calls)
    const [appliedSubtaskTypes, setAppliedSubtaskTypes] = useState<string[]>([]);
    const [appliedBatches, setAppliedBatches] = useState<string[]>([]);
    const [appliedTaskBatches, setAppliedTaskBatches] = useState<string[]>([]);

    const [taskCounts, setTaskCounts] = useState<StatusCount[]>([]);
    const [taskChartLoading, setTaskChartLoading] = useState(false);
    const [subtaskCounts, setSubtaskCounts] = useState<StatusCount[]>([]);
    const [subtaskChartLoading, setSubtaskChartLoading] = useState(false);

    // Fetch subtask types when project changes
    useEffect(() => {
        const fetchSubtaskTypes = async () => {
            if (!projectId) return;

            setSubtaskTypesLoading(true);
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
                setSubtaskTypesLoading(false);
            }
        };

        fetchSubtaskTypes();
    }, [projectId]);

    // Fetch initial chart data
    useEffect(() => {
        if (projectId) {
            fetchChartData();
        }
    }, [projectId]);

    const fetchChartData = async () => {
        if (!projectId) return;

        setTaskChartLoading(true);
        setSubtaskChartLoading(true);
        try {
            const [taskStatusCounts, subtaskStatusCounts] = await Promise.all([
                firebaseService.getTaskStatusCounts(projectId),
                firebaseService.getSubtaskStatusCounts(projectId)
            ]);

            setTaskCounts(taskStatusCounts);
            setSubtaskCounts(subtaskStatusCounts);
        } catch (error) {
            console.error('Error fetching chart data:', error);
        } finally {
            setTaskChartLoading(false);
            setSubtaskChartLoading(false);
        }
    };

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

    const applyFilters = async () => {
        if (!projectId) return;

        setSubtaskChartLoading(true);
        try {
            const subtaskStatusCounts = await firebaseService.getSubtaskStatusCounts(
                projectId,
                selectedSubtaskTypes
            );
            setSubtaskCounts(subtaskStatusCounts);
            setAppliedSubtaskTypes(selectedSubtaskTypes);
            setAppliedBatches(selectedBatches);
        } catch (error) {
            console.error('Error applying subtask filters:', error);
        } finally {
            setSubtaskChartLoading(false);
        }
    };

    const applyTaskFilters = async () => {
        if (!projectId) return;

        setTaskChartLoading(true);
        try {
            const taskStatusCounts = await firebaseService.getTaskStatusCounts(projectId);
            setTaskCounts(taskStatusCounts);
        } catch (error) {
            console.error('Error applying task filters:', error);
        } finally {
            setTaskChartLoading(false);
        }
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

    // Prepare chart data
    const taskChartData: ChartData = {
        labels: taskCounts.map(count => taskStatusLabels[count.status as keyof typeof taskStatusLabels] || count.status),
        datasets: [{
            data: taskCounts.map(count => count.count),
            backgroundColor: taskCounts.map(count => taskStatusColors[count.status as keyof typeof taskStatusColors] || '#999'),
            borderColor: taskCounts.map(count => taskStatusColors[count.status as keyof typeof taskStatusColors] || '#999'),
            borderWidth: 1
        }]
    };

    const subtaskChartData: ChartData = {
        labels: subtaskCounts.map(count => subtaskStatusLabels[count.status as keyof typeof subtaskStatusLabels] || count.status),
        datasets: [{
            data: subtaskCounts.map(count => count.count),
            backgroundColor: subtaskCounts.map(count => subtaskStatusColors[count.status as keyof typeof subtaskStatusColors] || '#999'),
            borderColor: subtaskCounts.map(count => subtaskStatusColors[count.status as keyof typeof subtaskStatusColors] || '#999'),
            borderWidth: 1
        }]
    };

    // Prepare type distribution chart data
    const typeDistributionData: ChartData = {
        labels: subtaskTypeInfos
            .filter(t => appliedSubtaskTypes.includes(t.type) || appliedSubtaskTypes.length === 0)
            .map(info => info.type),
        datasets: [{
            data: subtaskTypeInfos
                .filter(t => appliedSubtaskTypes.includes(t.type) || appliedSubtaskTypes.length === 0)
                .map(info => info.count),
            backgroundColor: subtaskTypeInfos
                .filter(t => appliedSubtaskTypes.includes(t.type) || appliedSubtaskTypes.length === 0)
                .map((_, index) => typeColors[index % typeColors.length]),
            borderColor: subtaskTypeInfos
                .filter(t => appliedSubtaskTypes.includes(t.type) || appliedSubtaskTypes.length === 0)
                .map((_, index) => typeColors[index % typeColors.length]),
            borderWidth: 1
        }]
    };

    // Prepare batch distribution chart data
    const batchDistributionData: ChartData = {
        labels: ['Batch 1', 'Batch 2'],
        datasets: [{
            data: [30, 70],
            backgroundColor: [typeColors[2], typeColors[0]], // Orange and Blue
            borderColor: [typeColors[2], typeColors[0]],
            borderWidth: 1
        }]
    };

    if (!projectId) {
        return (
            <Box sx={{ p: 3 }}>
                <Paper sx={{ p: 3 }}>
                    <Typography>Please select a project to view progress.</Typography>
                </Paper>
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

                        {/* Task Charts */}
                        <Box sx={{
                            minHeight: 340,
                            position: 'relative',
                            mt: 4
                        }}>
                            {taskChartLoading ? (
                                <Box sx={{
                                    display: 'flex',
                                    justifyContent: 'center',
                                    alignItems: 'center',
                                    position: 'absolute',
                                    top: 0,
                                    left: 0,
                                    right: 0,
                                    bottom: 0,
                                    backgroundColor: 'rgba(255, 255, 255, 0.8)',
                                    zIndex: 10
                                }}>
                                    <CircularProgress size={40} />
                                </Box>
                            ) : taskCounts.length > 0 && taskCounts.some(item => item.count > 0) ? (
                                <Box sx={{
                                    display: 'flex',
                                    flexWrap: 'wrap',
                                    justifyContent: 'space-between',
                                    gap: 2
                                }}>
                                    <Box sx={{
                                        width: { xs: '100%', md: 'calc(50% - 16px)' },
                                        mb: { xs: 4, md: 0 }
                                    }}>
                                        <GenericPieChart
                                            data={taskChartData}
                                            title="Task Status Distribution"
                                        />
                                    </Box>
                                    <Box sx={{
                                        width: { xs: '100%', md: 'calc(50% - 16px)' }
                                    }}>
                                        <GenericPieChart
                                            data={batchDistributionData}
                                            title="Task Batch Distribution"
                                        />
                                    </Box>
                                </Box>
                            ) : (
                                <NoDataDisplay type="tasks" />
                            )}
                        </Box>
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

                        {/* Subtask Charts */}
                        <Box sx={{
                            minHeight: 340,
                            position: 'relative',
                            mt: 4
                        }}>
                            {subtaskChartLoading ? (
                                <Box sx={{
                                    display: 'flex',
                                    justifyContent: 'center',
                                    alignItems: 'center',
                                    position: 'absolute',
                                    top: 0,
                                    left: 0,
                                    right: 0,
                                    bottom: 0,
                                    backgroundColor: 'rgba(255, 255, 255, 0.8)',
                                    zIndex: 10
                                }}>
                                    <CircularProgress size={40} />
                                </Box>
                            ) : subtaskCounts.length > 0 && subtaskCounts.some(item => item.count > 0) ? (
                                <Box sx={{
                                    display: 'flex',
                                    flexWrap: 'wrap',
                                    justifyContent: 'space-between',
                                    gap: 2
                                }}>
                                    <Box sx={{
                                        width: { xs: '100%', md: 'calc(33% - 16px)' },
                                        mb: { xs: 4, md: 0 }
                                    }}>
                                        <GenericPieChart
                                            data={subtaskChartData}
                                            title="Subtask Status Distribution"
                                        />
                                    </Box>
                                    <Box sx={{
                                        width: { xs: '100%', md: 'calc(33% - 16px)' },
                                        mb: { xs: 4, md: 0 }
                                    }}>
                                        <GenericPieChart
                                            data={typeDistributionData}
                                            title="Subtask Type Distribution"
                                        />
                                    </Box>
                                    <Box sx={{
                                        width: { xs: '100%', md: 'calc(33% - 16px)' }
                                    }}>
                                        <GenericPieChart
                                            data={batchDistributionData}
                                            title="Batch Distribution"
                                        />
                                    </Box>
                                </Box>
                            ) : (
                                <NoDataDisplay type="subtasks" />
                            )}
                        </Box>
                    </>
                )}
            </Paper>
        </Box>
    );
} 