import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/firebase';

export interface TimeEntry {
    created_ts: number;
    duration_seconds: number;
    last_updated_ts: number;
    subtask_id: string;
    user_id: string;
}

export async function GET(request: NextRequest) {
    try {
        // Get query parameters
        const searchParams = request.nextUrl.searchParams;
        const projectId = searchParams.get('projectId');
        const subtaskId = searchParams.get('subtaskId');

        if (!projectId) {
            return NextResponse.json(
                { error: 'Project ID is required' },
                { status: 400 }
            );
        }

        if (!subtaskId) {
            return NextResponse.json(
                { error: 'Subtask ID is required' },
                { status: 400 }
            );
        }

        console.log(`API: Fetching time entries for subtask: ${subtaskId} in project: ${projectId}`);

        // Reference to the timesheets collection for this project
        const timesheetsRef = db.collection(`projects/${projectId}/timesheets`);

        // Query for time entries with the given subtask_id
        const query = timesheetsRef.where('subtask_id', '==', subtaskId);

        // Execute the query
        const timesheetsSnapshot = await query.get();

        if (timesheetsSnapshot.empty) {
            console.log(`API: No time entries found for subtask: ${subtaskId}`);
            return NextResponse.json({
                timeEntries: []
            });
        }

        // Convert to array and format data
        const timeEntries = timesheetsSnapshot.docs.map(doc => {
            const data = doc.data();
            return {
                entry_id: doc.id,
                ...data,
            };
        });

        console.log(`API: Found ${timeEntries.length} time entries for subtask: ${subtaskId}`);
        return NextResponse.json({
            timeEntries
        });
    } catch (error: any) {
        console.error('API Error fetching time entries:', error);
        return NextResponse.json(
            { error: 'Failed to fetch time entries', details: error.message },
            { status: 500 }
        );
    }
} 