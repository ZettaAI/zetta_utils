import { ChartData } from '@/services/types';

// Status color mapping
export const taskStatusColors = {
    ingested: 'rgba(54, 162, 235, 0.8)',        // Blue
    pending_ingestion: 'rgba(255, 206, 86, 0.8)', // Yellow
    fully_processed: 'rgba(75, 192, 192, 0.8)'    // Green
} as const;

// Status label mapping for tasks
export const taskStatusLabels = {
    ingested: 'Ingested',
    pending_ingestion: 'Pending Ingestion',
    fully_processed: 'Fully Processed'
} as const;

// Status color mapping for subtasks
export const subtaskStatusColors = {
    fully_processed: 'rgba(75, 192, 192, 0.8)',    // Green
    pending_ingestion: 'rgba(255, 99, 132, 0.8)'   // Red
} as const;

// Status label mapping for subtasks
export const subtaskStatusLabels = {
    fully_processed: 'Completed',
    pending_ingestion: 'Pending'
} as const;

// Type color palette for dynamic assignment
const colorPalette = [
    'rgba(54, 162, 235, 0.8)',  // Blue
    'rgba(255, 99, 132, 0.8)',  // Red
    'rgba(75, 192, 192, 0.8)',  // Green
    'rgba(255, 159, 64, 0.8)',  // Orange
    'rgba(153, 102, 255, 0.8)', // Purple
    'rgba(255, 206, 86, 0.8)',  // Yellow
    'rgba(231, 233, 237, 0.8)'  // Grey
] as const;

export const getTypeColors = (types: string[]): Record<string, string> => {
    return types.reduce((acc, type, index) => {
        acc[type] = colorPalette[index % colorPalette.length];
        return acc;
    }, {} as Record<string, string>);
};

export const defaultChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: {
        legend: {
            position: 'right' as const,
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
    }
}; 