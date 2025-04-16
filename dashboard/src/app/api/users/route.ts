import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/firebase';

export interface User {
    user_id: string;
    hourly_rate: number;
    active_subtask: string;
    qualified_subtask_types?: string[];
    last_active_ts?: number;
}

export async function GET(request: NextRequest) {
    try {
        // Get query parameters
        const searchParams = request.nextUrl.searchParams;
        const projectId = searchParams.get('projectId');
        const page = parseInt(searchParams.get('page') || '1');
        const pageSize = parseInt(searchParams.get('pageSize') || '10');
        const sortBy = searchParams.get('sortBy') || 'user_id';
        const sortOrder = searchParams.get('sortOrder') || 'asc';

        if (!projectId) {
            return NextResponse.json(
                { error: 'Project ID is required' },
                { status: 400 }
            );
        }

        console.log(`API: Fetching users for project: ${projectId}`);

        // Reference to the users collection for this project
        const usersRef = db.collection(`projects/${projectId}/users`);

        // Get total count first
        const totalSnapshot = await usersRef.count().get();
        const total = totalSnapshot.data().count;

        // Calculate pagination
        const offset = (page - 1) * pageSize;

        // Create query with sorting and pagination
        let query = usersRef.orderBy(sortBy, sortOrder as 'asc' | 'desc')
            .offset(offset)
            .limit(pageSize);

        const usersSnapshot = await query.get();

        if (usersSnapshot.empty) {
            console.log(`API: No users found for project: ${projectId}`);
            return NextResponse.json({
                users: [],
                total,
                page,
                pageSize,
                totalPages: Math.ceil(total / pageSize)
            });
        }

        // Convert to array and format data
        const users = usersSnapshot.docs.map(doc => {
            const data = doc.data();
            return {
                user_id: doc.id,
                ...data,
            };
        });

        console.log(`API: Found ${users.length} users`);
        return NextResponse.json({
            users,
            total,
            page,
            pageSize,
            totalPages: Math.ceil(total / pageSize)
        });
    } catch (error: any) {
        console.error('API Error fetching users:', error);
        return NextResponse.json(
            { error: 'Failed to fetch users', details: error.message },
            { status: 500 }
        );
    }
} 