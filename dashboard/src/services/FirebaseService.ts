// Task status types
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

// Get a single task for testing
export async function getSingleTask(projectId: string): Promise<Task | null> {
    console.log(`📥 Fetching a single task from project: ${projectId} via API`);
    try {
        const response = await fetch(`/api/tasks?projectId=${projectId}`);

        if (!response.ok) {
            const errorData = await response.json();
            console.error('❌ API error response:', errorData);
            throw new Error(`API error: ${errorData.error || response.statusText}`);
        }

        const data = await response.json();
        console.log('📄 API response:', data);

        if (!data.tasks || data.tasks.length === 0) {
            console.log('❌ No tasks found in the collection');
            return null;
        }

        // Return the first task
        const task = data.tasks[0];
        console.log('📄 Found task:', JSON.stringify(task, null, 2));

        return task;
    } catch (error) {
        console.error('❌ Error fetching single task:', error);
        if (error instanceof Error) {
            console.error('❌ Error message:', error.message);
            console.error('❌ Error stack:', error.stack);
        }
        return null;
    }
}

// Get counts of tasks by status
export async function getTaskStatusCounts(projectId: string): Promise<StatusCount[]> {
    console.log(`📊 Fetching task status counts for project: ${projectId} via API`);

    try {
        const response = await fetch(`/api/tasks/status-counts?projectId=${projectId}`);

        if (!response.ok) {
            const errorData = await response.json();
            console.error('❌ API error response:', errorData);
            throw new Error(`API error: ${errorData.error || response.statusText}`);
        }

        const data = await response.json();
        console.log('📊 API response:', data);

        return data;
    } catch (error) {
        console.error('❌ Error getting task status counts:', error);
        if (error instanceof Error) {
            console.error('❌ Error message:', error.message);
            console.error('❌ Error stack:', error.stack);
        }

        // Return fallback counts to avoid infinite loading
        const statuses: TaskStatus[] = ['ingested', 'pending_ingestion', 'fully_processed'];
        return statuses.map(status => ({ status, count: 0 }));
    }
}

// Get subtask types with counts for a project
export async function getSubtaskTypesWithCounts(projectId: string): Promise<SubtaskTypeInfo[]> {
    console.log(`📋 Fetching subtask types with counts for project: ${projectId} via API`);

    try {
        const response = await fetch(`/api/subtasks/types-with-counts?projectId=${projectId}`);

        if (!response.ok) {
            const errorData = await response.json();
            console.error('❌ API error response:', errorData);
            throw new Error(`API error: ${errorData.error || response.statusText}`);
        }

        const data = await response.json();
        console.log('📋 Subtask types with counts:', data.types);

        return data.types || [];
    } catch (error) {
        console.error('❌ Error getting subtask types with counts:', error);
        if (error instanceof Error) {
            console.error('❌ Error message:', error.message);
            console.error('❌ Error stack:', error.stack);
        }

        return [];
    }
}

// Get subtask types for a project
export async function getSubtaskTypes(projectId: string): Promise<string[]> {
    console.log(`📋 Fetching subtask types for project: ${projectId} via API`);

    try {
        const response = await fetch(`/api/subtasks/types?projectId=${projectId}`);

        if (!response.ok) {
            const errorData = await response.json();
            console.error('❌ API error response:', errorData);
            throw new Error(`API error: ${errorData.error || response.statusText}`);
        }

        const data = await response.json();
        console.log('📋 Subtask types:', data.types);

        return data.types || [];
    } catch (error) {
        console.error('❌ Error getting subtask types:', error);
        if (error instanceof Error) {
            console.error('❌ Error message:', error.message);
            console.error('❌ Error stack:', error.stack);
        }

        return [];
    }
}

// Get subtask status counts
export async function getSubtaskStatusCounts(projectId: string, subtaskTypes?: string[]): Promise<StatusCount[]> {
    const typesDesc = subtaskTypes && subtaskTypes.length > 0
        ? `, types: [${subtaskTypes.join(', ')}]`
        : '';
    console.log(`📊 Fetching subtask status counts for project: ${projectId}${typesDesc} via API`);

    try {
        const url = new URL(`/api/subtasks/status-counts`, window.location.origin);
        url.searchParams.append('projectId', projectId);

        if (subtaskTypes && subtaskTypes.length > 0) {
            subtaskTypes.forEach(type => {
                url.searchParams.append('subtaskType', type);
            });
        }

        const response = await fetch(url.toString());

        if (!response.ok) {
            const errorData = await response.json();
            console.error('❌ API error response:', errorData);
            throw new Error(`API error: ${errorData.error || response.statusText}`);
        }

        const data = await response.json();
        console.log('📊 Subtask status counts:', data);

        return data;
    } catch (error) {
        console.error('❌ Error getting subtask status counts:', error);
        if (error instanceof Error) {
            console.error('❌ Error message:', error.message);
            console.error('❌ Error stack:', error.stack);
        }

        // Return fallback counts to avoid infinite loading
        const statuses: TaskStatus[] = ['pending_ingestion', 'fully_processed'];
        return statuses.map(status => ({ status, count: 0 }));
    }
} 