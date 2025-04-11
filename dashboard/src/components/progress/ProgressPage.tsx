'use client';

import { Box, Paper, Tabs, Tab, Alert, CircularProgress } from '@mui/material';
import { useSearchParams } from 'next/navigation';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, Title } from 'chart.js';
import { TabPanel } from '@/components/shared/TabPanel';
import { StatusPieChart } from './charts/StatusPieChart';
import { TypeDistributionChart } from './charts/TypeDistributionChart';
import { FilterBox } from './filters/FilterBox';
import { FilterDropdown } from './filters/FilterDropdown';
import { useFilters } from '@/hooks/useFilters';
import { useProjectData } from '@/hooks/useProjectData';

// Register ChartJS components
ChartJS.register(ArcElement, Tooltip, Legend, Title);

export default function ProgressPage() {
    const searchParams = useSearchParams();
    const projectId = searchParams.get('project') || '';
    const tabParam = searchParams.get('tab') || 'tasks';
    const tabValue = tabParam === 'subtasks' ? 1 : 0;

    const {
        taskCounts,
        subtaskCounts,
        subtaskTypeInfos,
        error,
        isLoading,
        fetchFilteredData
    } = useProjectData(projectId);

    const {
        selectedFilters: selectedSubtaskTypes,
        appliedFilters: appliedSubtaskTypes,
        handleFilterChange: handleSubtaskTypeChange,
        handleFilterRemove: handleSubtaskTypeRemove,
        handleFilterClear: handleSubtaskTypeClear,
        handleFilterApply: handleSubtaskTypeApply
    } = useFilters(subtaskTypeInfos.map(info => ({ type: 'type', value: info.type })));

    const {
        selectedFilters: selectedTaskBatches,
        appliedFilters: appliedTaskBatches,
        handleFilterChange: handleTaskBatchChange,
        handleFilterRemove: handleTaskBatchRemove,
        handleFilterClear: handleTaskBatchClear,
        handleFilterApply: handleTaskBatchApply
    } = useFilters([]);

    if (error) {
        return <Alert severity="error">{error}</Alert>;
    }

    if (isLoading) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                <CircularProgress />
            </Box>
        );
    }

    return (
        <Box sx={{ width: '100%' }}>
            <Paper sx={{ width: '100%', mb: 2 }}>
                <Tabs
                    value={tabValue}
                    onChange={(_, newValue) => {
                        const newTab = newValue === 1 ? 'subtasks' : 'tasks';
                        const params = new URLSearchParams(searchParams);
                        params.set('tab', newTab);
                        window.history.pushState({}, '', `?${params.toString()}`);
                    }}
                    indicatorColor="primary"
                    textColor="primary"
                    sx={{ borderBottom: 1, borderColor: 'divider' }}
                >
                    <Tab label="Tasks" />
                    <Tab label="Subtasks" />
                </Tabs>
            </Paper>

            <TabPanel value={tabValue} index={0}>
                <FilterBox
                    filters={selectedTaskBatches}
                    onClear={handleTaskBatchClear}
                    onApply={handleTaskBatchApply}
                    onFilterRemove={handleTaskBatchRemove}
                >
                    <FilterDropdown
                        id="task-batch-filter"
                        label="Batch Filter"
                        values={[]}
                        selected={selectedTaskBatches.map(f => f.value)}
                        onChange={(e) => handleTaskBatchChange('batch', e.target.value)}
                    />
                </FilterBox>

                <StatusPieChart
                    data={taskCounts}
                    title="Task Status Distribution"
                    isSubtask={false}
                />
            </TabPanel>

            <TabPanel value={tabValue} index={1}>
                <FilterBox
                    filters={selectedSubtaskTypes}
                    onClear={handleSubtaskTypeClear}
                    onApply={handleSubtaskTypeApply}
                    onFilterRemove={handleSubtaskTypeRemove}
                >
                    <FilterDropdown
                        id="subtask-type-filter"
                        label="Type Filter"
                        values={subtaskTypeInfos.map(info => ({
                            value: info.type,
                            label: info.type
                        }))}
                        selected={selectedSubtaskTypes.map(f => f.value)}
                        onChange={(e) => handleSubtaskTypeChange('type', e.target.value)}
                    />
                </FilterBox>

                <Box sx={{ display: 'flex', gap: 2 }}>
                    <StatusPieChart
                        data={subtaskCounts}
                        title="Subtask Status Distribution"
                        isSubtask={true}
                    />
                    <TypeDistributionChart
                        typeInfos={subtaskTypeInfos}
                        title="Subtask Type Distribution"
                    />
                </Box>
            </TabPanel>
        </Box>
    );
} 