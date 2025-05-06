import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/firebase';
import { Query, DocumentData } from 'firebase-admin/firestore';

export interface Subtask {
    subtask_id: string;
    task_id: string;
    subtask_type: string;
    is_active: boolean;
    assigned_user_id: string;
    active_user_id: string;
    completed_user_id: string;
    completion_status: string;
    priority: number;
    batch_id: string;
    last_leased_ts: number;
    ng_state: string;
    ng_state_initial: string;
}

export async function GET(request: NextRequest) {
    try {
        // Get query parameters
        const searchParams = request.nextUrl.searchParams;
        const projectId = searchParams.get('projectId');
        const page = parseInt(searchParams.get('page') || '1');
        const pageSize = parseInt(searchParams.get('pageSize') || '10');
        const sortBy = searchParams.get('sortBy') || 'subtask_id';
        const sortOrder = searchParams.get('sortOrder') || 'asc';

        // Filters
        const subtaskType = searchParams.get('subtaskType');
        const isActive = searchParams.get('isActive');
        const isComplete = searchParams.get('isComplete');
        const assignedUserId = searchParams.get('assignedUserId');
        const activeUserId = searchParams.get('activeUserId');
        const completedUserId = searchParams.get('completedUserId');
        const taskId = searchParams.get('taskId');
        const batchId = searchParams.get('batchId');
        const completionStatus = searchParams.get('completionStatus');
        const lastLeasedTsStart = searchParams.get('lastLeasedTsStart');
        const lastLeasedTsEnd = searchParams.get('lastLeasedTsEnd');

        if (!projectId) {
            return NextResponse.json(
                { error: 'Project ID is required' },
                { status: 400 }
            );
        }

        console.log(`API: Fetching subtasks for project: ${projectId}`);

        // Reference to the subtasks collection for this project
        const subtasksRef = db.collection(`projects/${projectId}/subtasks`);

        // Build the query with filters
        let firestoreQuery: Query<DocumentData> = subtasksRef;

        // Apply filters
        if (subtaskType) {
            firestoreQuery = firestoreQuery.where('subtask_type', '==', subtaskType);
        }

        if (isActive !== null) {
            firestoreQuery = firestoreQuery.where('is_active', '==', isActive === 'true');
        }

        // Handle is_complete filter by filtering on completion_status with only equality operator
        if (isComplete !== null) {
            // If isComplete is true, filter for non-empty completion_status
            if (isComplete === 'true') {
                firestoreQuery = firestoreQuery.where('completion_status', '!=', '');
            }
            // For false case, we'll handle in the frontend since we can't use multiple != operators
        }

        if (assignedUserId) {
            firestoreQuery = firestoreQuery.where('assigned_user_id', '==', assignedUserId);
        }

        if (activeUserId) {
            firestoreQuery = firestoreQuery.where('active_user_id', '==', activeUserId);
        }

        if (completedUserId) {
            firestoreQuery = firestoreQuery.where('completed_user_id', '==', completedUserId);
        }

        if (taskId) {
            firestoreQuery = firestoreQuery.where('task_id', '==', taskId);
        }

        if (batchId) {
            firestoreQuery = firestoreQuery.where('batch_id', '==', batchId);
        }

        if (completionStatus) {
            firestoreQuery = firestoreQuery.where('completion_status', '==', completionStatus);
        }

        // For timestamp range, we need to handle differently
        if (lastLeasedTsStart && lastLeasedTsEnd) {
            firestoreQuery = firestoreQuery.where('last_leased_ts', '>=', parseFloat(lastLeasedTsStart))
                .where('last_leased_ts', '<=', parseFloat(lastLeasedTsEnd));
        } else if (lastLeasedTsStart) {
            firestoreQuery = firestoreQuery.where('last_leased_ts', '>=', parseFloat(lastLeasedTsStart));
        } else if (lastLeasedTsEnd) {
            firestoreQuery = firestoreQuery.where('last_leased_ts', '<=', parseFloat(lastLeasedTsEnd));
        }

        // Add sorting
        firestoreQuery = firestoreQuery.orderBy(sortBy, sortOrder as 'asc' | 'desc');

        // Get total count
        const countSnapshot = await firestoreQuery.count().get();
        const total = countSnapshot.data().count;

        // Add pagination
        firestoreQuery = firestoreQuery.offset((page - 1) * pageSize).limit(pageSize);

        // Execute the query
        const subtasksSnapshot = await firestoreQuery.get();

        if (subtasksSnapshot.empty) {
            console.log(`API: No subtasks found for project: ${projectId}`);
            return NextResponse.json({
                subtasks: [],
                total,
                page,
                pageSize,
                totalPages: Math.ceil(total / pageSize)
            });
        }

        // Convert to array and format data
        const subtasks = subtasksSnapshot.docs.map(doc => {
            const data = doc.data();
            return {
                subtask_id: doc.id,
                ...data,
            };
        });

        console.log(`API: Found ${subtasks.length} subtasks`);
        return NextResponse.json({
            subtasks,
            total,
            page,
            pageSize,
            totalPages: Math.ceil(total / pageSize)
        });
    } catch (error: any) {
        console.error('API Error fetching subtasks:', error);
        return NextResponse.json(
            { error: 'Failed to fetch subtasks', details: error.message },
            { status: 500 }
        );
    }
} 