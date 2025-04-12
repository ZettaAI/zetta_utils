import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/firebase';

export interface SubtaskType {
    subtask_type: string;
    completion_statuses: string[];
    description?: string;
}

export async function GET(request: NextRequest) {
    try {
        // Get query parameters
        const searchParams = request.nextUrl.searchParams;
        const projectId = searchParams.get('projectId');

        if (!projectId) {
            return NextResponse.json(
                { error: 'Project ID is required' },
                { status: 400 }
            );
        }

        console.log(`API: Fetching subtask types for project: ${projectId}`);

        // Reference to the subtask_types collection for this project
        const subtaskTypesRef = db.collection(`projects/${projectId}/subtask_types`);
        const subtaskTypesSnapshot = await subtaskTypesRef.get();

        if (subtaskTypesSnapshot.empty) {
            console.log(`API: No subtask types found for project: ${projectId}`);
            return NextResponse.json({ subtaskTypes: [] });
        }

        // Convert to array and format data
        const subtaskTypes = subtaskTypesSnapshot.docs.map(doc => {
            const data = doc.data();
            return {
                subtask_type: doc.id,
                ...data,
            };
        });

        console.log(`API: Found ${subtaskTypes.length} subtask types`);
        return NextResponse.json({ subtaskTypes });
    } catch (error: any) {
        console.error('API Error fetching subtask types:', error);
        return NextResponse.json(
            { error: 'Failed to fetch subtask types', details: error.message },
            { status: 500 }
        );
    }
} 