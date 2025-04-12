'use client';

import { useState, useEffect } from 'react';
import { Box, FormControl, Select, MenuItem, CircularProgress, Alert, SelectChangeEvent } from '@mui/material';
import { useRouter, useSearchParams } from 'next/navigation';

interface Project {
    id: string;
}

export default function ProjectSelector() {
    const [projects, setProjects] = useState<Project[]>([]);
    const [selectedProject, setSelectedProject] = useState<string>('');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const router = useRouter();
    const searchParams = useSearchParams();

    useEffect(() => {
        const fetchProjects = async () => {
            try {
                const response = await fetch('/api/projects');
                if (!response.ok) {
                    throw new Error('Failed to fetch projects');
                }
                const data = await response.json();
                setProjects(data);
                const currentProjectId = searchParams.get('projectId');
                if (currentProjectId) {
                    setSelectedProject(currentProjectId);
                } else if (data.length > 0) {
                    setSelectedProject(data[0].id);
                    // Only set URL if no project is selected
                    const params = new URLSearchParams(window.location.search);
                    params.set('projectId', data[0].id);
                    const newUrl = `${window.location.pathname}?${params.toString()}`;
                    router.replace(newUrl);
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

    const handleProjectChange = (event: SelectChangeEvent<string>) => {
        const projectId = event.target.value;
        setSelectedProject(projectId);
        const params = new URLSearchParams(window.location.search);
        params.set('projectId', projectId);
        const newUrl = `${window.location.pathname}?${params.toString()}`;
        router.replace(newUrl);
    };

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
                        <FormControl sx={{ minWidth: 300 }}>
                            <Select
                                value={selectedProject}
                                onChange={handleProjectChange}
                                size="small"
                                sx={{
                                    backgroundColor: 'white',
                                    '& .MuiOutlinedInput-notchedOutline': {
                                        borderColor: 'rgb(209, 213, 219)'
                                    }
                                }}
                            >
                                {projects.map((project) => (
                                    <MenuItem key={project.id} value={project.id}>
                                        {project.id}
                                    </MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                    </div>
                </div>
            </div>
        </div>
    );
} 