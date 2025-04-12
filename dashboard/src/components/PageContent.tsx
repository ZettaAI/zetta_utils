'use client';

import { Box } from '@mui/material';
import { useDashboard } from './layout/DashboardLayout';
import ProgressPage from './progress/ProgressPage';
import ProgressNewPage from './progress/ProgressNewPage';
import UsersPage from './users/UsersPage';

export default function PageContent() {
    const { currentPage, currentProject } = useDashboard();

    if (!currentProject) {
        return null; // Don't render anything until project is selected
    }

    const renderPageContent = () => {
        // If currentPage is empty (initial load), don't show any page
        if (!currentPage) {
            return null;
        }

        switch (currentPage) {
            case 'progress':
                return <ProgressPage />;
            case 'progress-new':
                return <ProgressNewPage />;
            case 'insights':
                return (
                    <Box sx={{ p: 3 }}>
                        <h1>Insights Page</h1>
                        <p>Project: {currentProject}</p>
                    </Box>
                );
            case 'tasks':
                return (
                    <Box sx={{ p: 3 }}>
                        <h1>Tasks Page</h1>
                        <p>Project: {currentProject}</p>
                    </Box>
                );
            case 'subtasks':
                return (
                    <Box sx={{ p: 3 }}>
                        <h1>Subtasks Page</h1>
                        <p>Project: {currentProject}</p>
                    </Box>
                );
            case 'users':
                return <UsersPage />;
            default:
                return (
                    <Box sx={{ p: 3 }}>
                        <h1>Page Not Found</h1>
                        <p>The requested page "{currentPage}" does not exist.</p>
                    </Box>
                );
        }
    };

    return (
        <Box sx={{ p: 3 }}>
            {renderPageContent()}
        </Box>
    );
} 