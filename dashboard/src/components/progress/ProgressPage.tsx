'use client';

import { useState, useEffect } from 'react';
import { useRouter, useSearchParams, usePathname } from 'next/navigation';
import { Box, Paper, Tabs, Tab, Typography, CircularProgress, Alert, FormControl, Select, MenuItem, InputLabel, Chip, OutlinedInput, Checkbox, ListItemText, Button } from '@mui/material';
import { Pie } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, Title } from 'chart.js';
import firebaseService from '@/services/FirebaseService';
import { StatusCount, SubtaskTypeInfo } from '@/services/types';
import { FilterBox } from './filters/FilterBox';
import { FilterDropdown } from './filters/FilterDropdown';

// Register ChartJS components
ChartJS.register(ArcElement, Tooltip, Legend, Title);

// Status color mapping
const statusColors = {
    ingested: 'rgba(54, 162, 235, 0.8)',        // Blue
    pending_ingestion: 'rgba(255, 206, 86, 0.8)', // Yellow
    fully_processed: 'rgba(75, 192, 192, 0.8)'    // Green
};

// Status label mapping - using exact status names
const statusLabels = {
    ingested: 'ingested',
    pending_ingestion: 'pending_ingestion',
    fully_processed: 'fully_processed'
};

// Status color mapping for subtasks
const subtaskStatusColors = {
    fully_processed: 'rgba(75, 192, 192, 0.8)',    // Green
    pending_ingestion: 'rgba(255, 99, 132, 0.8)'   // Red
};

// Status label mapping for subtasks
const subtaskStatusLabels = {
    fully_processed: 'Completed',
    pending_ingestion: 'Pending'
};

// Type color mapping - dynamically assigned
const getTypeColors = (types: string[]) => {
    const colorPalette = [
        'rgba(54, 162, 235, 0.8)',  // Blue
        'rgba(255, 99, 132, 0.8)',  // Red
        'rgba(75, 192, 192, 0.8)',  // Green
        'rgba(255, 159, 64, 0.8)',  // Orange
        'rgba(153, 102, 255, 0.8)', // Purple
        'rgba(255, 206, 86, 0.8)',  // Yellow
        'rgba(231, 233, 237, 0.8)'  // Grey
    ];

    return types.reduce((acc, type, index) => {
        acc[type] = colorPalette[index % colorPalette.length];
        return acc;
    }, {} as Record<string, string>);
};

// Batch color mapping - for dummy data
const batchColors = {
    'batch_1': 'rgba(255, 99, 132, 0.8)',   // Red
    'batch_2': 'rgba(54, 162, 235, 0.8)',   // Blue
    'batch_3': 'rgba(255, 206, 86, 0.8)'    // Yellow
};

// Batch options for both tabs
const batchOptions = [
    { value: 'batch_1', label: 'batch_1' },
    { value: 'batch_2', label: 'batch_2' }
];

interface TabPanelProps {
    children?: React.ReactNode;
    index: number;
    value: number;
}

function TabPanel(props: TabPanelProps) {
    const { children, value, index, ...other } = props;

    return (
        <div
            role="tabpanel"
            hidden={value !== index}
            id={`simple-tabpanel-${index}`}
            aria-labelledby={`simple-tab-${index}`}
            {...other}
        >
            {value === index && (
                <Box sx={{ p: 3 }}>
                    {children}
                </Box>
            )}
        </div>
    );
}

function StatusPieChart({ data, title, isSubtask = false }: { data: StatusCount[], title: string, isSubtask?: boolean }) {
    // Define the status mappings with explicit types
    const taskStatusColors: Record<string, string> = {
        ingested: 'rgba(54, 162, 235, 0.8)',        // Blue
        pending_ingestion: 'rgba(255, 206, 86, 0.8)', // Yellow
        fully_processed: 'rgba(75, 192, 192, 0.8)'    // Green
    };

    const taskStatusLabels: Record<string, string> = {
        ingested: 'Ingested',
        pending_ingestion: 'Pending Ingestion',
        fully_processed: 'Fully Processed'
    };

    const subtaskStatusColors: Record<string, string> = {
        fully_processed: 'rgba(75, 192, 192, 0.8)',    // Green
        pending_ingestion: 'rgba(255, 99, 132, 0.8)'   // Red
    };

    const subtaskStatusLabels: Record<string, string> = {
        fully_processed: 'Completed',
        pending_ingestion: 'Pending'
    };

    // Choose the correct status mappings based on whether it's for subtasks
    const statusColors = isSubtask ? subtaskStatusColors : taskStatusColors;
    const statusLabels = isSubtask ? subtaskStatusLabels : taskStatusLabels;

    // Log the data to debug
    console.log('StatusPieChart data:', data);
    console.log('StatusPieChart isSubtask:', isSubtask);

    // Create chart data from status counts
    const chartData = {
        labels: data.map(item => {
            const label = statusLabels[item.status];
            if (!label) console.warn('Missing label for status:', item.status);
            return label;
        }),
        datasets: [
            {
                data: data.map(item => item.count),
                backgroundColor: data.map(item => {
                    const color = statusColors[item.status];
                    if (!color) console.warn('Missing color for status:', item.status);
                    return color;
                }),
                borderWidth: 1,
            },
        ],
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
            legend: {
                position: 'right' as const,
            },
            title: {
                display: true,
                text: title,
                font: {
                    size: 16
                }
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
        <Box sx={{ height: 300, position: 'relative' }}>
            <Pie data={chartData} options={options} />
        </Box>
    );
}

function TypeDistributionChart({ typeInfos, title }: { typeInfos: SubtaskTypeInfo[], title: string }) {
    const types = typeInfos.map(info => info.type);
    const typeColors = getTypeColors(types);

    const data = {
        labels: types,
        datasets: [
            {
                data: typeInfos.map(info => info.count),
                backgroundColor: types.map(type => typeColors[type]),
                borderWidth: 1,
            },
        ],
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
            legend: {
                position: 'right' as const,
            },
            title: {
                display: true,
                text: title,
                font: {
                    size: 16
                }
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
        <Box sx={{ height: 300, position: 'relative' }}>
            <Pie data={data} options={options} />
        </Box>
    );
}

function BatchDistributionChart({ title }: { title: string }) {
    // Dummy data for batch distribution
    const dummyBatchData = [
        { batch: 'batch_1', count: 65 },
        { batch: 'batch_2', count: 30 },
        { batch: 'batch_3', count: 5 }
    ];

    const data = {
        labels: dummyBatchData.map(item => item.batch),
        datasets: [
            {
                data: dummyBatchData.map(item => item.count),
                backgroundColor: dummyBatchData.map(item => batchColors[item.batch as keyof typeof batchColors]),
                borderWidth: 1,
            },
        ],
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
            legend: {
                position: 'right' as const,
            },
            title: {
                display: true,
                text: title,
                font: {
                    size: 16
                }
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
        <Box sx={{ height: 300, position: 'relative' }}>
            <Pie data={data} options={options} />
        </Box>
    );
}

// A simple component for when we have no data
function NoDataDisplay({ type }: { type: string }) {
    const isSubtasks = type === 'subtasks';

    return (
        <Box sx={{ p: 4, textAlign: 'center', height: 300, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <Typography variant="body1" color="text.secondary">
                No {type} data found. This could be because:
            </Typography>
            <ul style={{ textAlign: 'left', maxWidth: '400px', margin: '1rem auto' }}>
                <li>There are no {type} in this project</li>
                {isSubtasks && <li>There are no active subtasks (is_active=true)</li>}
                <li>Database connection is not properly configured</li>
                <li>You don't have sufficient permissions</li>
            </ul>
        </Box>
    );
}

export default function ProgressPage() {
    const router = useRouter();
    const pathname = usePathname();
    const searchParams = useSearchParams();
    const projectId = searchParams.get('project') || '';
    const urlTab = searchParams.get('tab') || 'tasks';

    const [tabValue, setTabValue] = useState(urlTab === 'subtasks' ? 1 : 0);
    const [taskCounts, setTaskCounts] = useState<StatusCount[]>([]);
    const [subtaskCounts, setSubtaskCounts] = useState<StatusCount[]>([]);
    const [subtaskTypeInfos, setSubtaskTypeInfos] = useState<SubtaskTypeInfo[]>([]);

    // States for selected filters (pending application)
    const [selectedSubtaskTypes, setSelectedSubtaskTypes] = useState<string[]>([]);
    const [selectedBatches, setSelectedBatches] = useState<string[]>([]);

    // Task filters
    const [selectedTaskBatches, setSelectedTaskBatches] = useState<string[]>([]);

    // States for applied filters (used in API calls)
    const [appliedSubtaskTypes, setAppliedSubtaskTypes] = useState<string[]>([]);
    const [appliedBatches, setAppliedBatches] = useState<string[]>([]);
    const [appliedTaskBatches, setAppliedTaskBatches] = useState<string[]>([]);

    // Remove the main loading state and only keep chart-specific loading states
    const [chartLoading, setChartLoading] = useState(false);
    const [taskChartLoading, setTaskChartLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Fetch subtask types and initial data when project changes
    useEffect(() => {
        let isMounted = true;

        async function fetchInitialData() {
            if (!projectId) {
                setTaskChartLoading(false);
                setChartLoading(false);
                return;
            }

            try {
                setTaskChartLoading(true);
                console.log('Fetching task counts...');
                const initialTaskCounts = await firebaseService.getTaskStatusCounts(projectId);
                if (isMounted) {
                    console.log('Setting task counts:', initialTaskCounts);
                    setTaskCounts(initialTaskCounts);
                    setTaskChartLoading(false);
                }

                // Fetch subtask types
                const typeInfos = await firebaseService.getSubtaskTypesWithCounts(projectId);
                if (!isMounted) return;

                setSubtaskTypeInfos(typeInfos);
                const allTypes = typeInfos.map(info => info.type);

                // Only set types if none are selected
                if (selectedSubtaskTypes.length === 0) {
                    setSelectedSubtaskTypes(allTypes);
                    setAppliedSubtaskTypes(allTypes);
                }

                // Then fetch task and subtask counts
                const [updatedTaskCounts, subtasks] = await Promise.all([
                    firebaseService.getTaskStatusCounts(projectId),
                    firebaseService.getSubtaskStatusCounts(projectId, allTypes)
                ]);

                if (!isMounted) return;

                setTaskCounts(updatedTaskCounts);
                setSubtaskCounts(subtasks);
            } catch (error) {
                console.error('Error in ProgressPage fetchInitialData:', error);
                if (isMounted) {
                    setError(
                        error instanceof Error
                            ? error.message
                            : 'An unexpected error occurred while fetching data'
                    );
                }
            } finally {
                if (isMounted) {
                    setTaskChartLoading(false);
                    setChartLoading(false);
                }
            }
        }

        fetchInitialData();

        return () => {
            isMounted = false;
        };
    }, [projectId]); // Only re-run when projectId changes

    const handleTaskBatchChange = (event: any) => {
        const {
            target: { value },
        } = event;

        const batches = typeof value === 'string' ? value.split(',') : value;
        setSelectedTaskBatches(batches);
    };

    const handleBatchChange = (event: any) => {
        const {
            target: { value },
        } = event;

        const batches = typeof value === 'string' ? value.split(',') : value;
        setSelectedBatches(batches);
    };

    const handleSubtaskTypeChange = (event: any) => {
        const {
            target: { value },
        } = event;

        const types = typeof value === 'string' ? value.split(',') : value;
        setSelectedSubtaskTypes(types);
    };

    // Initial data load
    useEffect(() => {
        let isMounted = true;
        async function fetchData() {
            if (!projectId) return;

            try {
                // Start loading only when actually fetching
                setTaskChartLoading(true);
                console.log('Fetching task counts...');
                const taskData = await firebaseService.getTaskStatusCounts(projectId);
                if (isMounted) {
                    console.log('Setting task counts:', taskData);
                    setTaskCounts(taskData);
                    setTaskChartLoading(false);
                }

                // Fetch subtask status counts separately
                setChartLoading(true);
                console.log(`Fetching subtask counts for types: ${appliedSubtaskTypes.join(', ')}...`);
                const subtaskData = await firebaseService.getSubtaskStatusCounts(projectId, appliedSubtaskTypes);
                if (isMounted) {
                    console.log('Setting subtask counts:', subtaskData);
                    setSubtaskCounts(subtaskData);
                    setChartLoading(false);
                }
            } catch (error) {
                console.error('Error in ProgressPage fetchData:', error);
                if (isMounted) {
                    setError(
                        error instanceof Error
                            ? error.message
                            : 'An unexpected error occurred while fetching data'
                    );
                }
            } finally {
                if (isMounted) {
                    setTaskChartLoading(false);
                    setChartLoading(false);
                }
            }
        }

        fetchData();

        return () => {
            isMounted = false;
        };
    }, [projectId]);

    // Function to fetch only subtask data when filters are applied
    const fetchSubtaskDataWithFilters = async () => {
        if (!projectId) return;
        setChartLoading(true);
        try {
            const filteredSubtaskData = await firebaseService.getSubtaskStatusCounts(
                projectId,
                appliedSubtaskTypes
            );
            setSubtaskCounts(filteredSubtaskData);
        } catch (error) {
            console.error('Error fetching filtered subtask data:', error);
            setError(error instanceof Error ? error.message : 'Error fetching subtask data');
        } finally {
            setChartLoading(false);
        }
    };

    // Function to fetch only task data when filters are applied
    const fetchTaskDataWithFilters = async () => {
        if (!projectId) return;
        setTaskChartLoading(true);
        try {
            const filteredTaskData = await firebaseService.getTaskStatusCounts(projectId);
            setTaskCounts(filteredTaskData);
        } catch (error) {
            console.error('Error fetching filtered task data:', error);
            setError(error instanceof Error ? error.message : 'Error fetching task data');
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
            // Handle task batch filter removal
            setSelectedTaskBatches(prev => prev.filter(b => b !== value));
        } else if (tabValue === 1) {
            // Handle subtask filters removal
            if (type === 'type') {
                setSelectedSubtaskTypes(prev => prev.filter(t => t !== value));
            } else if (type === 'batch') {
                setSelectedBatches(prev => prev.filter(b => b !== value));
            }
        }
    };

    const applyFilters = () => {
        setAppliedSubtaskTypes(selectedSubtaskTypes);
        setAppliedBatches(selectedBatches);
        fetchSubtaskDataWithFilters();
    };

    const applyTaskFilters = () => {
        setAppliedTaskBatches(selectedTaskBatches);
        fetchTaskDataWithFilters();
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
                <Alert severity="error">
                    {error}
                </Alert>
            </Box>
        );
    }

    if (taskChartLoading || chartLoading) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
                <CircularProgress />
            </Box>
        );
    }

    return (
        <Box sx={{ width: '100%' }}>
            <Paper sx={{ width: '100%', mb: 2 }}>
                <Tabs
                    value={tabValue}
                    onChange={(_, newValue) => setTabValue(newValue)}
                    indicatorColor="primary"
                    textColor="primary"
                    sx={{ borderBottom: 1, borderColor: 'divider' }}
                >
                    <Tab label="Tasks" />
                    <Tab label="Subtasks" />
                </Tabs>

                {error && (
                    <Alert severity="error" sx={{ m: 2 }}>
                        {error}
                    </Alert>
                )}

                <Box sx={{ position: 'relative' }}>
                    <TabPanel value={tabValue} index={0}>
                        {/* Always show Task Filters */}
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
                            position: 'relative'
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
                            ) : (
                                taskCounts.length > 0 && taskCounts.some(item => item.count > 0) ? (
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
                                            <StatusPieChart data={taskCounts} title="Task Status Distribution" />
                                        </Box>
                                        <Box sx={{
                                            width: { xs: '100%', md: 'calc(50% - 16px)' }
                                        }}>
                                            <BatchDistributionChart title="Task Batch Distribution" />
                                        </Box>
                                    </Box>
                                ) : (
                                    <NoDataDisplay type="tasks" />
                                )
                            )}
                        </Box>
                    </TabPanel>

                    <TabPanel value={tabValue} index={1}>
                        {/* Always show Subtask Filters */}
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
                            position: 'relative'
                        }}>
                            {chartLoading ? (
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
                            ) : (
                                subtaskCounts.length > 0 && subtaskCounts.some(item => item.count > 0) ? (
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
                                            <StatusPieChart data={subtaskCounts} title="Subtask Status Distribution" isSubtask={true} />
                                        </Box>
                                        <Box sx={{
                                            width: { xs: '100%', md: 'calc(33% - 16px)' },
                                            mb: { xs: 4, md: 0 }
                                        }}>
                                            <TypeDistributionChart
                                                typeInfos={subtaskTypeInfos.filter(t =>
                                                    appliedSubtaskTypes.includes(t.type) || appliedSubtaskTypes.length === 0
                                                )}
                                                title="Subtask Type Distribution"
                                            />
                                        </Box>
                                        <Box sx={{
                                            width: { xs: '100%', md: 'calc(33% - 16px)' }
                                        }}>
                                            <BatchDistributionChart title="Batch Distribution" />
                                        </Box>
                                    </Box>
                                ) : (
                                    <NoDataDisplay type="subtasks" />
                                )
                            )}
                        </Box>
                    </TabPanel>
                </Box>
            </Paper>
        </Box>
    );
} 