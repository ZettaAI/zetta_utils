import { CollectionReference, DocumentData, Query, QueryDocumentSnapshot, getFirestore, collection, query, where, getDocs, Firestore } from 'firebase/firestore';
import { initializeApp, getApps, FirebaseApp } from 'firebase/app';
import { FilterOption, StatusCount } from './types';

// Firebase configuration
const firebaseConfig = {
    apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
    authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
    projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
    storageBucket: process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET,
    messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID,
    appId: process.env.NEXT_PUBLIC_FIREBASE_APP_ID
};

// Task status types
export type TaskStatus = 'ingested' | 'pending_ingestion' | 'fully_processed';

export interface SubtaskTypeInfo {
    type: string;
    count: number;
}

export interface Task {
    task_id: string;
    status: TaskStatus;
    [key: string]: any; // Allow for other properties
}

export class FirebaseService {
    private static instance: FirebaseService;
    private app: FirebaseApp;
    private db: Firestore;

    private constructor() {
        try {
            // Initialize Firebase only if no apps exist
            this.app = getApps().length === 0 ? initializeApp(firebaseConfig) : getApps()[0];
            this.db = getFirestore(this.app);
            console.log('✅ Firebase initialized successfully');
        } catch (error) {
            console.error('❌ Error initializing Firebase:', error);
            throw error;
        }
    }

    public static getInstance(): FirebaseService {
        if (!FirebaseService.instance) {
            FirebaseService.instance = new FirebaseService();
        }
        return FirebaseService.instance;
    }

    // Get a single task for testing
    async getSingleTask(projectId: string): Promise<Task | null> {
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
            return null;
        }
    }

    // Get counts of tasks by status
    async getTaskStatusCounts(projectId: string): Promise<StatusCount[]> {
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
            // Return fallback counts to avoid infinite loading
            const statuses: TaskStatus[] = ['ingested', 'pending_ingestion', 'fully_processed'];
            return statuses.map(status => ({ status, count: 0 }));
        }
    }

    // Get subtask types with counts for a project
    async getSubtaskTypesWithCounts(projectId: string): Promise<SubtaskTypeInfo[]> {
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
            return [];
        }
    }

    // Get subtask types for a project
    async getSubtaskTypes(projectId: string): Promise<string[]> {
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
            return [];
        }
    }

    // Get subtask status counts
    async getSubtaskStatusCounts(projectId: string, subtaskTypes?: string[]): Promise<StatusCount[]> {
        const typesDesc = subtaskTypes && subtaskTypes.length > 0
            ? `, types: [${subtaskTypes.join(', ')}]`
            : '';
        console.log(`📊 Fetching subtask status counts for project: ${projectId}${typesDesc} via API`);

        try {
            let url = `/api/subtasks/status-counts?projectId=${projectId}`;
            if (subtaskTypes && subtaskTypes.length > 0) {
                url += `&types=${encodeURIComponent(subtaskTypes.join(','))}`;
            }
            const response = await fetch(url);

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
            // Return fallback counts to avoid infinite loading
            const statuses: TaskStatus[] = ['ingested', 'pending_ingestion', 'fully_processed'];
            return statuses.map(status => ({ status, count: 0 }));
        }
    }

    async getBatchDistribution(projectId: string, filters: FilterOption[] = []): Promise<{ batchName: string; count: number; }[]> {
        try {
            const subtasksRef = collection(this.db, 'projects', projectId, 'subtasks') as CollectionReference<DocumentData>;
            let q = query(subtasksRef);

            // Apply filters if any
            filters.forEach(filter => {
                if (filter.type && filter.value) {
                    q = query(q, where(filter.type, '==', filter.value));
                }
            });

            const snapshot = await getDocs(q);
            const batchCounts = new Map<string, number>();

            snapshot.forEach((doc: QueryDocumentSnapshot) => {
                const data = doc.data();
                const batchName = data.batchName || 'Unassigned';
                batchCounts.set(batchName, (batchCounts.get(batchName) || 0) + 1);
            });

            return Array.from(batchCounts.entries()).map(([batchName, count]) => ({
                batchName,
                count
            }));
        } catch (error) {
            console.error('Error fetching batch distribution:', error);
            return [];
        }
    }
}

// Create a singleton instance
const firebaseService = FirebaseService.getInstance();

// Export the class and create a singleton instance
export default firebaseService;