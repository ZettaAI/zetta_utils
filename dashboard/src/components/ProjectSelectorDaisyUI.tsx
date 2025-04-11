'use client';

import { useState, useEffect } from 'react';

interface Project {
    id: string;
}

export default function ProjectSelectorDaisyUI() {
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
        <div className="flex justify-center p-4">
            <span className="loading loading-spinner loading-lg text-primary"></span>
        </div>
    );

    if (error) return (
        <div className="alert alert-error shadow-lg m-4">
            <svg xmlns="http://www.w3.org/2000/svg" className="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>{error}</span>
        </div>
    );

    return (
        <div className="navbar bg-base-100 fixed top-0 left-0 right-0 shadow-md">
            <div className="max-w-7xl mx-auto w-full px-4">
                <div className="flex-1">
                    <div className="form-control w-full max-w-xs">
                        <label className="label">
                            <span className="label-text">Project</span>
                        </label>
                        <select
                            className="select select-bordered w-full max-w-xs"
                            value={selectedProject}
                            onChange={(e) => setSelectedProject(e.target.value)}
                        >
                            <option disabled value="">Select a project</option>
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