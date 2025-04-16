import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/firebase';

interface UserBasic {
    user_id: string;
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

        console.log(`API: Fetching users list for project: ${projectId}`);

        // Reference to the users collection for this project
        const usersRef = db.collection(`projects/${projectId}/users`);

        // Only select user_id field to keep it lightweight
        const usersSnapshot = await usersRef.select('user_id').get();

        if (usersSnapshot.empty) {
            console.log(`API: No users found for project: ${projectId}`);
            return NextResponse.json({ users: [] });
        }

        // Convert to array and format data
        const users = usersSnapshot.docs.map(doc => ({
            user_id: doc.id,
        }));

        console.log(`API: Found ${users.length} users for dropdown`);
        return NextResponse.json({ users });
    } catch (error: any) {
        console.error('API Error fetching users list:', error);
        return NextResponse.json(
            { error: 'Failed to fetch users list', details: error.message },
            { status: 500 }
        );
    }
} 