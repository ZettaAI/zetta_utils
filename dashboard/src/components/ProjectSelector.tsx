'use client';

import { useState, useEffect } from 'react';

interface Project {
    id: string;
}

export default function ProjectSelector() {
    const [projects, setProjects] = useState<Project[]>([]);
    const [selectedProject, setSelectedProject] = useState<string>('');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchProjects = async () => {
            try {
                const response = await fetch('/api/projects');
                if (!response.ok) {
                    throw new Error('Failed to fetch projects');
                }
                const data = await response.json();
                setProjects(data);
                if (data.length > 0) {
                    setSelectedProject(data[0].id);
                }
            } catch (err) {
                console.error('Error fetching projects:', err);
                setError('Failed to fetch projects. Please check your GCP credentials and try again.');
            } finally {
                setLoading(false);
            }
        };

        fetchProjects();
    }, []);

    if (loading) return (
        <div className="fixed top-0 left-0 right-0 bg-white border-b shadow-md">
            <div className="max-w-7xl mx-auto px-4">
                <div className="h-16 flex items-center gap-4">
                    <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-gray-600">Project:</span>
                        <div className="w-64 h-10 bg-gray-100 rounded animate-pulse" />
                    </div>
                </div>
            </div>
        </div>
    );
    if (error) return (
        <div className="fixed top-0 left-0 right-0 bg-white border-b shadow-md">
            <div className="max-w-7xl mx-auto px-4">
                <div className="h-16 flex items-center gap-4">
                    <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-gray-600">Project:</span>
                        <div className="text-red-500 text-sm w-64">{error}</div>
                    </div>
                </div>
            </div>
        </div>
    );

    return (
        <div className="fixed top-0 left-0 right-0 bg-white border-b shadow-md">
            <div className="max-w-7xl mx-auto px-4">
                <div className="h-16 flex items-center gap-4">
                    <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-gray-600">Project:</span>
                        <select
                            value={selectedProject}
                            onChange={(e) => setSelectedProject(e.target.value)}
                            className="w-64 h-10 border border-gray-300 rounded-md shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition ease-in-out duration-150"
                        >
                            <option value="" disabled>Select a project</option>
                            {projects.map((project) => (
                                <option key={project.id} value={project.id}>
                                    {project.id}
                                </option>
                            ))}
                        </select>
                    </div>
                </div>
            </div>
        </div>
    );
} 