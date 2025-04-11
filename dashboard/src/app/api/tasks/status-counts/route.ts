import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/firebase';

// Task status types
export type TaskStatus = 'ingested' | 'pending_ingestion' | 'fully_processed';

export interface StatusCount {
    status: TaskStatus;
    count: number;
}

export async function GET(request: NextRequest) {
    try {
        // Get the projectId from the query parameter
        const searchParams = request.nextUrl.searchParams;
        const projectId = searchParams.get('projectId');

        if (!projectId) {
            return NextResponse.json(
                { error: 'Project ID is required' },
                { status: 400 }
            );
        }

        console.log(`API: Fetching task status counts for project: ${projectId}`);

        const statuses: TaskStatus[] = ['ingested', 'pending_ingestion', 'fully_processed'];
        const result: StatusCount[] = [];

        // Reference to the tasks collection for this project
        const tasksRef = db.collection(`projects/${projectId}/tasks`);

        // For each status, get count
        for (const status of statuses) {
            try {
                const statusQuery = tasksRef.where('status', '==', status);
                const querySnapshot = await statusQuery.get();

                console.log(`API: Found ${querySnapshot.docs.length} tasks with status: ${status}`);
                result.push({
                    status,
                    count: querySnapshot.docs.length
                });
            } catch (queryError) {
                console.error(`API: Error querying for status ${status}:`, queryError);
                result.push({ status, count: 0 });
            }
        }

        console.log(`API: Final status counts:`, result);
        return NextResponse.json(result);
    } catch (error: any) {
        console.error('API Error fetching task status counts:', error);
        return NextResponse.json(
            { error: 'Failed to fetch task status counts', details: error.message },
            { status: 500 }
        );
    }
} 