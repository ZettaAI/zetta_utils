'use client';

import { useState, useEffect } from 'react';
import {
    Box,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
    CircularProgress,
    Alert,
    Paper,
    IconButton,
    Drawer,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
    Divider,
    Typography
} from '@mui/material';
import { useRouter, useSearchParams } from 'next/navigation';
import MenuIcon from '@mui/icons-material/Menu';
import InsightsIcon from '@mui/icons-material/Insights';
import BarChartIcon from '@mui/icons-material/BarChart';
import TaskIcon from '@mui/icons-material/Task';
import FormatListBulletedIcon from '@mui/icons-material/FormatListBulleted';
import PeopleIcon from '@mui/icons-material/People';

interface Project {
    id: string;
}

type PageType = 'progress' | 'insights' | 'tasks' | 'subtasks' | 'users' | 'progress-new';

export default function ProjectSelectorMUI() {
    const router = useRouter();
    const searchParams = useSearchParams();

    const [projects, setProjects] = useState<Project[]>([]);
    const [selectedProject, setSelectedProject] = useState<string>('');
    const [currentPage, setCurrentPage] = useState<PageType>('progress');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [drawerOpen, setDrawerOpen] = useState(false);

    // Read project and page from URL on initial load
    useEffect(() => {
        const projectFromUrl = searchParams.get('project');
        const pageFromUrl = searchParams.get('page') as PageType;

        if (projectFromUrl) {
            setSelectedProject(projectFromUrl);
        }

        if (pageFromUrl && ['progress', 'insights', 'tasks', 'subtasks', 'users', 'progress-new'].includes(pageFromUrl)) {
            setCurrentPage(pageFromUrl);
        } else {
            // Default to 'progress' if no valid page is in URL
            setCurrentPage('progress');
        }
    }, [searchParams]);

    useEffect(() => {
        const fetchProjects = async () => {
            try {
                const response = await fetch('/api/projects');
                if (!response.ok) {
                    throw new Error('Failed to fetch projects');
                }
                const data = await response.json();
                setProjects(data);

                // Only set from data if not already set from URL
                const projectFromUrl = searchParams.get('project');
                if (!projectFromUrl && data.length > 0) {
                    setSelectedProject(data[0].id);

                    // Update URL with the default project and current page
                    const pageFromUrl = searchParams.get('page') || 'progress';
                    router.push(`?project=${data[0].id}&page=${pageFromUrl}`);
                }
            } catch (err) {
                console.error('Error fetching projects:', err);
                setError('Failed to fetch projects. Please check your GCP credentials and try again.');
            } finally {
                setLoading(false);
            }
        };

        fetchProjects();
    }, [router, searchParams]);

    // Update URL when project selection changes
    const handleProjectChange = (value: string) => {
        setSelectedProject(value);

        // Preserve the current page when changing projects
        router.push(`?project=${value}&page=${currentPage}`);
    };

    // Navigate to a page
    const navigateToPage = (page: PageType) => {
        setCurrentPage(page);

        // Keep the current project when changing pages
        router.push(`?project=${selectedProject}&page=${page}`);

        // Close the drawer after navigation on mobile
        setDrawerOpen(false);
    };

    const toggleDrawer = (open: boolean) => {
        setDrawerOpen(open);
    };

    // Helper function to capitalize the first letter for display
    const formatPageName = (page: string): string => {
        return page.charAt(0).toUpperCase() + page.slice(1);
    };

    if (loading) return (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
            <CircularProgress />
        </Box>
    );

    if (error) return (
        <Alert severity="error" sx={{ m: 2 }}>
            {error}
        </Alert>
    );

    return (
        <>
            <Paper elevation={1} sx={{ position: 'fixed', top: 0, left: 0, right: 0, zIndex: 1000, borderRadius: 0 }}>
                <Box sx={{ maxWidth: '100%', mx: 'auto', px: 2 }}>
                    <Box sx={{ height: 72, display: 'flex', alignItems: 'center' }}>
                        <IconButton
                            edge="start"
                            color="inherit"
                            aria-label="menu"
                            onClick={() => toggleDrawer(true)}
                            sx={{ mr: 2 }}
                        >
                            <MenuIcon />
                        </IconButton>
                        <Box sx={{ display: 'flex', alignItems: 'center', mr: 4 }}>
                            <Typography
                                variant="h5"
                                sx={{
                                    fontWeight: 500,
                                    color: '#5f6368',
                                    mr: 2,
                                    fontSize: '1.4rem'
                                }}
                            >
                                {formatPageName(currentPage)}
                            </Typography>
                        </Box>
                        <FormControl sx={{ minWidth: 200 }}>
                            <InputLabel id="project-select-label">Project</InputLabel>
                            <Select
                                labelId="project-select-label"
                                value={selectedProject}
                                label="Project"
                                onChange={(e) => handleProjectChange(e.target.value as string)}
                                sx={{
                                    minWidth: 200,
                                    height: 40,
                                    '& .MuiSelect-select': {
                                        paddingTop: '8px',
                                        paddingBottom: '8px'
                                    }
                                }}
                                size="small"
                            >
                                {projects.map((project) => (
                                    <MenuItem key={project.id} value={project.id}>
                                        {project.id}
                                    </MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                    </Box>
                </Box>
            </Paper>

            <Drawer
                anchor="left"
                open={drawerOpen}
                onClose={() => toggleDrawer(false)}
                PaperProps={{
                    sx: { width: 280 }
                }}
            >
                <Box sx={{ pt: 2, pb: 2, px: 2 }}>
                    <Typography variant="h6" sx={{ fontWeight: 500, color: '#5f6368' }}>
                        Task Dashboard
                    </Typography>
                </Box>
                <Divider />
                <List>
                    <ListItem
                        onClick={() => navigateToPage('progress')}
                        sx={{
                            cursor: 'pointer',
                            bgcolor: currentPage === 'progress' ? 'rgba(0, 0, 0, 0.04)' : 'transparent',
                            '&:hover': { bgcolor: 'rgba(0, 0, 0, 0.08)' }
                        }}
                    >
                        <ListItemIcon sx={{ color: currentPage === 'progress' ? 'primary.main' : 'inherit' }}>
                            <BarChartIcon />
                        </ListItemIcon>
                        <ListItemText
                            primary="Progress"
                            primaryTypographyProps={{
                                color: currentPage === 'progress' ? 'primary' : 'inherit',
                                fontWeight: currentPage === 'progress' ? 500 : 400
                            }}
                        />
                    </ListItem>
                    <ListItem
                        onClick={() => navigateToPage('insights')}
                        sx={{
                            cursor: 'pointer',
                            bgcolor: currentPage === 'insights' ? 'rgba(0, 0, 0, 0.04)' : 'transparent',
                            '&:hover': { bgcolor: 'rgba(0, 0, 0, 0.08)' }
                        }}
                    >
                        <ListItemIcon sx={{ color: currentPage === 'insights' ? 'primary.main' : 'inherit' }}>
                            <InsightsIcon />
                        </ListItemIcon>
                        <ListItemText
                            primary="Insights"
                            primaryTypographyProps={{
                                color: currentPage === 'insights' ? 'primary' : 'inherit',
                                fontWeight: currentPage === 'insights' ? 500 : 400
                            }}
                        />
                    </ListItem>
                    <ListItem
                        onClick={() => navigateToPage('tasks')}
                        sx={{
                            cursor: 'pointer',
                            bgcolor: currentPage === 'tasks' ? 'rgba(0, 0, 0, 0.04)' : 'transparent',
                            '&:hover': { bgcolor: 'rgba(0, 0, 0, 0.08)' }
                        }}
                    >
                        <ListItemIcon sx={{ color: currentPage === 'tasks' ? 'primary.main' : 'inherit' }}>
                            <TaskIcon />
                        </ListItemIcon>
                        <ListItemText
                            primary="Tasks"
                            primaryTypographyProps={{
                                color: currentPage === 'tasks' ? 'primary' : 'inherit',
                                fontWeight: currentPage === 'tasks' ? 500 : 400
                            }}
                        />
                    </ListItem>
                    <ListItem
                        onClick={() => navigateToPage('subtasks')}
                        sx={{
                            cursor: 'pointer',
                            bgcolor: currentPage === 'subtasks' ? 'rgba(0, 0, 0, 0.04)' : 'transparent',
                            '&:hover': { bgcolor: 'rgba(0, 0, 0, 0.08)' }
                        }}
                    >
                        <ListItemIcon sx={{ color: currentPage === 'subtasks' ? 'primary.main' : 'inherit' }}>
                            <FormatListBulletedIcon />
                        </ListItemIcon>
                        <ListItemText
                            primary="Subtasks"
                            primaryTypographyProps={{
                                color: currentPage === 'subtasks' ? 'primary' : 'inherit',
                                fontWeight: currentPage === 'subtasks' ? 500 : 400
                            }}
                        />
                    </ListItem>
                    <ListItem
                        onClick={() => navigateToPage('users')}
                        sx={{
                            cursor: 'pointer',
                            bgcolor: currentPage === 'users' ? 'rgba(0, 0, 0, 0.04)' : 'transparent',
                            '&:hover': { bgcolor: 'rgba(0, 0, 0, 0.08)' }
                        }}
                    >
                        <ListItemIcon sx={{ color: currentPage === 'users' ? 'primary.main' : 'inherit' }}>
                            <PeopleIcon />
                        </ListItemIcon>
                        <ListItemText
                            primary="Users"
                            primaryTypographyProps={{
                                color: currentPage === 'users' ? 'primary' : 'inherit',
                                fontWeight: currentPage === 'users' ? 500 : 400
                            }}
                        />
                    </ListItem>
                    <ListItem
                        onClick={() => navigateToPage('progress-new')}
                        sx={{
                            cursor: 'pointer',
                            bgcolor: currentPage === 'progress-new' ? 'rgba(0, 0, 0, 0.04)' : 'transparent',
                            '&:hover': { bgcolor: 'rgba(0, 0, 0, 0.08)' }
                        }}
                    >
                        <ListItemIcon sx={{ color: currentPage === 'progress-new' ? 'primary.main' : 'inherit' }}>
                            <BarChartIcon />
                        </ListItemIcon>
                        <ListItemText
                            primary="Progress New"
                            primaryTypographyProps={{
                                color: currentPage === 'progress-new' ? 'primary' : 'inherit',
                                fontWeight: currentPage === 'progress-new' ? 500 : 400
                            }}
                        />
                    </ListItem>
                </List>
            </Drawer>
        </>
    );
} 