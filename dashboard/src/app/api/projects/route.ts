import { NextResponse } from 'next/server';
import { db } from '@/lib/firebase';

export async function GET() {
    try {
        // First, let's verify we can access the database
        console.log('Attempting to fetch projects...');
        const projectsSnapshot = await db.collection('projects').get();

        // Just return the document IDs
        const projects = projectsSnapshot.docs.map(doc => ({
            id: doc.id
        }));

        console.log('Found projects:', projects);
        return NextResponse.json(projects);
    } catch (error: any) {
        console.error('Error fetching projects:', error);
        return NextResponse.json(
            { error: 'Failed to fetch projects', details: error.message },
            { status: 500 }
        );
    }
} 