import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/firebase';

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

        console.log(`API: Fetching subtask types for project: ${projectId}`);

        // Reference to the subtask_types subcollection for this project
        const subtaskTypesRef = db.collection(`projects/${projectId}/subtask_types`);
        const subtaskTypesSnapshot = await subtaskTypesRef.get();

        if (subtaskTypesSnapshot.empty) {
            console.log(`API: No subtask types found for project: ${projectId}`);
            return NextResponse.json({ types: [] });
        }

        // Extract the subtask type IDs
        const types = subtaskTypesSnapshot.docs.map(doc => doc.id);

        console.log(`API: Found ${types.length} subtask types:`, types);
        return NextResponse.json({ types });
    } catch (error: any) {
        console.error('API Error fetching subtask types:', error);
        return NextResponse.json(
            { error: 'Failed to fetch subtask types', details: error.message },
            { status: 500 }
        );
    }
} 