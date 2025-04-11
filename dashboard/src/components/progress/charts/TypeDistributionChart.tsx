import { Box } from '@mui/material';
import { Pie } from 'react-chartjs-2';
import { SubtaskTypeInfo } from '@/services/types';
import { getTypeColors, defaultChartOptions } from '@/utils/chartConfig';

interface TypeDistributionChartProps {
    typeInfos: SubtaskTypeInfo[];
    title: string;
}

export function TypeDistributionChart({ typeInfos, title }: TypeDistributionChartProps) {
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
            <Pie data={data} options={options} />
        </Box>
    );
} 