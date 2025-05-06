import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/firebase';

export async function POST(request: NextRequest) {
    try {
        // Parse the request body
        const body = await request.json();
        const { projectId, userId } = body;

        if (!projectId) {
            return NextResponse.json(
                { error: 'Project ID is required' },
                { status: 400 }
            );
        }

        if (!userId) {
            return NextResponse.json(
                { error: 'User ID is required' },
                { status: 400 }
            );
        }

        console.log(`API: Deleting user ${userId} from project: ${projectId}`);

        // Reference to the user document
        const userDocRef = db.collection(`projects/${projectId}/users`).doc(userId);

        // Check if user exists
        const existingUser = await userDocRef.get();
        if (!existingUser.exists) {
            console.log(`API: User ${userId} does not exist in project ${projectId}`);
            return NextResponse.json(
                { error: 'User does not exist' },
                { status: 404 }
            );
        }

        // Check if user has an active subtask
        const userData = existingUser.data();
        if (userData && userData.active_subtask) {
            console.log(`API: Cannot delete user ${userId} with active subtask: ${userData.active_subtask}`);
            return NextResponse.json(
                { error: 'Cannot delete user with active subtask. Please reassign or complete their active task first.' },
                { status: 400 }
            );
        }

        // Delete the user
        await userDocRef.delete();

        console.log(`API: Successfully deleted user ${userId} from project ${projectId}`);
        return NextResponse.json({ success: true });
    } catch (error: any) {
        console.error('API Error deleting user:', error);
        return NextResponse.json(
            { error: 'Failed to delete user', details: error.message },
            { status: 500 }
        );
    }
} 