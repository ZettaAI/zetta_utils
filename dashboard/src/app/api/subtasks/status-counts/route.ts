import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/firebase';

// Task/subtask status types
export type TaskStatus = 'ingested' | 'pending_ingestion' | 'fully_processed';

export interface StatusCount {
    status: TaskStatus;
    count: number;
}

export async function GET(request: NextRequest) {
    try {
        // Get the parameters from the query
        const searchParams = request.nextUrl.searchParams;
        const projectId = searchParams.get('projectId');
        const subtaskTypes = searchParams.getAll('subtaskType');

        if (!projectId) {
            return NextResponse.json(
                { error: 'Project ID is required' },
                { status: 400 }
            );
        }

        console.log(`API: Fetching subtask status counts for project: ${projectId}${subtaskTypes.length > 0 ? `, filtered by types: ${subtaskTypes.join(', ')}` : ''}`);

        // Reference to the subtasks collection for this project
        const subtasksRef = db.collection(`projects/${projectId}/subtasks`);

        // Start with active subtasks filter
        let queryRef = subtasksRef.where('is_active', '==', true);

        // Add subtask type filter if provided
        if (subtaskTypes.length > 0 && !subtaskTypes.includes('all')) {
            // Unfortunately, Firestore doesn't support OR queries on multiple field values
            // We'll need to query each type separately and combine results
            let completedCount = 0;
            let incompleteCount = 0;

            for (const subtaskType of subtaskTypes) {
                const typeQuery = subtasksRef
                    .where('is_active', '==', true)
                    .where('subtask_type', '==', subtaskType);

                const snapshot = await typeQuery.get();

                // Count by completion status
                snapshot.forEach(doc => {
                    const data = doc.data();
                    if (data.completion_status && data.completion_status.trim() !== '') {
                        completedCount++;
                    } else {
                        incompleteCount++;
                    }
                });
            }

            // Return the combined counts
            const result = [
                { status: 'fully_processed' as TaskStatus, count: completedCount },
                { status: 'pending_ingestion' as TaskStatus, count: incompleteCount }
            ];

            console.log(`API: Subtask status counts for selected types:`, result);
            return NextResponse.json(result);
        }

        // If no specific types selected or "all" is included, query all active subtasks
        const activeSubtasksSnapshot = await queryRef.get();

        console.log(`API: Found ${activeSubtasksSnapshot.docs.length} active subtasks${subtaskTypes.length > 0 && !subtaskTypes.includes('all') ? ` of types: ${subtaskTypes.join(', ')}` : ''}`);

        // Count subtasks based on completion status
        let completedCount = 0;
        let incompleteCount = 0;

        activeSubtasksSnapshot.forEach(doc => {
            const data = doc.data();
            if (data.completion_status && data.completion_status.trim() !== '') {
                // Completed
                completedCount++;
            } else {
                // Incomplete
                incompleteCount++;
            }
        });

        // Map to our status types - note that we don't use 'ingested' status for subtasks
        const result = [
            { status: 'fully_processed' as TaskStatus, count: completedCount },
            { status: 'pending_ingestion' as TaskStatus, count: incompleteCount }
        ];

        console.log(`API: Subtask status counts:`, result);
        return NextResponse.json(result);
    } catch (error: any) {
        console.error('API Error fetching subtask status counts:', error);
        return NextResponse.json(
            { error: 'Failed to fetch subtask status counts', details: error.message },
            { status: 500 }
        );
    }
} 