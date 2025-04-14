import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/firebase';

export async function POST(request: NextRequest) {
    try {
        // Parse the request body
        const body = await request.json();
        const { projectId, userId, userData } = body;

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

        if (!userData) {
            return NextResponse.json(
                { error: 'User data is required' },
                { status: 400 }
            );
        }

        console.log(`API: Updating user ${userId} in project: ${projectId}`);

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

        // Update the user
        await userDocRef.update({
            ...userData,
            updated_at: Date.now()
        });

        console.log(`API: Successfully updated user ${userId} in project ${projectId}`);
        return NextResponse.json({ success: true, user_id: userId });
    } catch (error: any) {
        console.error('API Error updating user:', error);
        return NextResponse.json(
            { error: 'Failed to update user', details: error.message },
            { status: 500 }
        );
    }
} 