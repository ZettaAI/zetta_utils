import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/firebase';

export async function POST(request: NextRequest) {
    try {
        // Parse the request body
        const body = await request.json();
        const { projectId, userData } = body;

        if (!projectId) {
            return NextResponse.json(
                { error: 'Project ID is required' },
                { status: 400 }
            );
        }

        if (!userData || !userData.user_id) {
            return NextResponse.json(
                { error: 'User data and user_id are required' },
                { status: 400 }
            );
        }

        console.log(`API: Adding new user ${userData.user_id} to project: ${projectId}`);

        // Reference to the user document
        const userDocRef = db.collection(`projects/${projectId}/users`).doc(userData.user_id);

        // Check if user already exists
        const existingUser = await userDocRef.get();
        if (existingUser.exists) {
            console.log(`API: User ${userData.user_id} already exists in project ${projectId}`);
            return NextResponse.json(
                { error: 'User already exists' },
                { status: 409 }
            );
        }

        // Add the new user
        await userDocRef.set({
            user_id: userData.user_id,
            hourly_rate: userData.hourly_rate || 0,
            active_subtask: userData.active_subtask || "",
            qualified_subtask_types: userData.qualified_subtask_types || [],
            created_at: Date.now()
        });

        console.log(`API: Successfully added user ${userData.user_id} to project ${projectId}`);
        return NextResponse.json({ success: true, user_id: userData.user_id });
    } catch (error: any) {
        console.error('API Error adding user:', error);
        return NextResponse.json(
            { error: 'Failed to add user', details: error.message },
            { status: 500 }
        );
    }
} 