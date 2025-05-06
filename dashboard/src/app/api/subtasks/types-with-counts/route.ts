import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/firebase';

export interface SubtaskTypeInfo {
    type: string;
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

        console.log(`API: Fetching subtask types with counts for project: ${projectId}`);

        // Reference to the subtask_types subcollection for this project
        const subtaskTypesRef = db.collection(`projects/${projectId}/subtask_types`);
        const subtaskTypesSnapshot = await subtaskTypesRef.get();

        if (subtaskTypesSnapshot.empty) {
            console.log(`API: No subtask types found for project: ${projectId}`);
            return NextResponse.json({ types: [] });
        }

        // Extract the subtask type IDs
        const types = subtaskTypesSnapshot.docs.map(doc => doc.id);

        // Get counts for each subtask type (active only)
        const subtasksRef = db.collection(`projects/${projectId}/subtasks`);
        const result: SubtaskTypeInfo[] = [];

        // Getting counts for each subtask type
        for (const type of types) {
            const typeQuery = subtasksRef
                .where('is_active', '==', true)
                .where('subtask_type', '==', type);

            const snapshot = await typeQuery.get();
            result.push({
                type: type,
                count: snapshot.size
            });
        }

        // Sort by count in descending order
        result.sort((a, b) => b.count - a.count);

        console.log(`API: Found ${result.length} subtask types with counts:`, result);
        return NextResponse.json({ types: result });
    } catch (error: any) {
        console.error('API Error fetching subtask types with counts:', error);
        return NextResponse.json(
            { error: 'Failed to fetch subtask types with counts', details: error.message },
            { status: 500 }
        );
    }
} 