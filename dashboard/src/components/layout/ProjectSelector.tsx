import { useEffect, useState } from 'react';
import { FormControl, Select, MenuItem, InputLabel, Box, Skeleton } from '@mui/material';
import { useDashboard } from './DashboardLayout';

interface Project {
    id: string;
}

export default function ProjectSelector() {
    const { currentProject, setCurrentProject } = useDashboard();
    const [projects, setProjects] = useState<Project[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchProjects = async () => {
            try {
                const response = await fetch('/api/projects');
                if (!response.ok) {
                    throw new Error('Failed to fetch projects');
                }
                const data = await response.json();
                setProjects(data);

                // Set first project as default if none selected
                if (!currentProject && data.length > 0) {
                    setCurrentProject(data[0].id);
                }
            } catch (err) {
                console.error('Error fetching projects:', err);
            } finally {
                setLoading(false);
            }
        };

        fetchProjects();
    }, [currentProject, setCurrentProject]);

    const baseWidth = 200;

    return (
        <Box sx={{ width: baseWidth }}>
            <FormControl size="small" fullWidth>
                <InputLabel id="project-select-label">Project</InputLabel>
                {loading ? (
                    <Skeleton
                        variant="rectangular"
                        sx={{
                            height: 40,
                            borderRadius: 1,
                            bgcolor: 'grey.100',
                            transform: 'none'
                        }}
                    />
                ) : (
                    <Select
                        labelId="project-select-label"
                        value={currentProject || ''}
                        label="Project"
                        onChange={(e) => setCurrentProject(e.target.value)}
                    >
                        {projects.map((project) => (
                            <MenuItem key={project.id} value={project.id}>
                                {project.id}
                            </MenuItem>
                        ))}
                    </Select>
                )}
            </FormControl>
        </Box>
    );
} 