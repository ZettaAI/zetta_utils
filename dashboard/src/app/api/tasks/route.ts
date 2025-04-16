import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/firebase';

// Get a single task by project ID
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

        console.log(`API: Fetching tasks for project: ${projectId}`);

        // Reference to the tasks collection for this project
        const tasksRef = db.collection(`projects/${projectId}/tasks`);

        // Get all tasks (we'll implement pagination later if needed)
        const tasksSnapshot = await tasksRef.limit(10).get();

        if (tasksSnapshot.empty) {
            console.log(`API: No tasks found for project: ${projectId}`);
            return NextResponse.json({ tasks: [] });
        }

        // Convert to array and format data
        const tasks = tasksSnapshot.docs.map(doc => {
            const data = doc.data();
            return {
                task_id: doc.id,
                ...data,
            };
        });

        console.log(`API: Found ${tasks.length} tasks`);
        return NextResponse.json({ tasks });
    } catch (error: any) {
        console.error('API Error fetching tasks:', error);
        return NextResponse.json(
            { error: 'Failed to fetch tasks', details: error.message },
            { status: 500 }
        );
    }
} 