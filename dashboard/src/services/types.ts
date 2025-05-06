export type TaskStatus = 'ingested' | 'pending_ingestion' | 'fully_processed';

export interface StatusCount {
    status: TaskStatus;
    count: number;
}

export interface SubtaskTypeInfo {
    type: string;
    count: number;
}

export interface Task {
    task_id: string;
    status: TaskStatus;
    [key: string]: any; // Allow for other properties
}

export interface FilterOption {
    type: string;
    value: string;
}

export interface ChartData {
    labels: string[];
    datasets: {
        data: number[];
        backgroundColor: string[];
        borderWidth: number;
    }[];
} 