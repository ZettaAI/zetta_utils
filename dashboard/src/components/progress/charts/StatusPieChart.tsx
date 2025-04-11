import { Box } from '@mui/material';
import { Pie } from 'react-chartjs-2';
import { StatusCount } from '@/services/types';
import {
    taskStatusColors,
    taskStatusLabels,
    subtaskStatusColors,
    subtaskStatusLabels,
    defaultChartOptions
} from '@/utils/chartConfig';

interface StatusPieChartProps {
    data: StatusCount[];
    title: string;
    isSubtask?: boolean;
}

export function StatusPieChart({ data, title, isSubtask = false }: StatusPieChartProps) {
    // Choose the correct status mappings based on whether it's for subtasks
    const statusColors = isSubtask ? subtaskStatusColors : taskStatusColors;
    const statusLabels = isSubtask ? subtaskStatusLabels : taskStatusLabels;

    const chartData = {
        labels: data.map(item => {
            const label = statusLabels[item.status as keyof typeof statusLabels];
            if (!label) console.warn('Missing label for status:', item.status);
            return label;
        }),
        datasets: [
            {
                data: data.map(item => item.count),
                backgroundColor: data.map(item => {
                    const color = statusColors[item.status as keyof typeof statusColors];
                    if (!color) console.warn('Missing color for status:', item.status);
                    return color;
                }),
                borderWidth: 1,
            },
        ],
    };

    const options = {
        ...defaultChartOptions,
        plugins: {
            ...defaultChartOptions.plugins,
            title: {
                display: true,
                text: title,
                font: {
                    size: 16
                }
            }
        }
    };

    return (
        <Box sx={{ height: 300, position: 'relative' }}>
            <Pie data={chartData} options={options} />
        </Box>
    );
} 