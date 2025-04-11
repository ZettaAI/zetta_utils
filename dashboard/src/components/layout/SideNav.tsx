import { Drawer, List, ListItem, ListItemIcon, ListItemText, Box, Typography, Divider } from '@mui/material';
import BarChartIcon from '@mui/icons-material/BarChart';
import InsightsIcon from '@mui/icons-material/Insights';
import TaskIcon from '@mui/icons-material/Task';
import FormatListBulletedIcon from '@mui/icons-material/FormatListBulleted';
import PeopleIcon from '@mui/icons-material/People';
import { useDashboard } from './DashboardLayout';

interface NavItem {
    id: string;
    label: string;
    icon: JSX.Element;
}

const NAV_ITEMS: NavItem[] = [
    { id: 'progress', label: 'Progress', icon: <BarChartIcon /> },
    { id: 'insights', label: 'Insights', icon: <InsightsIcon /> },
    { id: 'tasks', label: 'Tasks', icon: <TaskIcon /> },
    { id: 'subtasks', label: 'Subtasks', icon: <FormatListBulletedIcon /> },
    { id: 'users', label: 'Users', icon: <PeopleIcon /> },
];

interface SideNavProps {
    open: boolean;
    onClose: () => void;
}

export default function SideNav({ open, onClose }: SideNavProps) {
    const { currentPage, setCurrentPage, currentProject } = useDashboard();

    const handleNavClick = (pageId: string) => {
        setCurrentPage(pageId);
        onClose();
    };

    return (
        <Drawer
            open={open}
            onClose={onClose}
            variant="temporary"
            PaperProps={{
                sx: { width: 280 }
            }}
        >
            <Box sx={{ pt: 2, pb: 2, px: 2 }}>
                <Typography variant="h6" sx={{ fontWeight: 500, color: 'grey.800' }}>
                    Task Dashboard
                </Typography>
                {currentProject && (
                    <Typography variant="body2" color="text.secondary">
                        Project: {currentProject}
                    </Typography>
                )}
            </Box>
            <Divider />
            <List>
                {NAV_ITEMS.map((item) => (
                    <ListItem
                        key={item.id}
                        onClick={() => handleNavClick(item.id)}
                        sx={{
                            cursor: 'pointer',
                            bgcolor: currentPage === item.id ? 'action.selected' : 'transparent',
                            '&:hover': { bgcolor: 'action.hover' }
                        }}
                    >
                        <ListItemIcon sx={{ color: currentPage === item.id ? 'primary.main' : 'inherit' }}>
                            {item.icon}
                        </ListItemIcon>
                        <ListItemText
                            primary={item.label}
                            primaryTypographyProps={{
                                color: currentPage === item.id ? 'primary' : 'inherit',
                                fontWeight: currentPage === item.id ? 500 : 400
                            }}
                        />
                    </ListItem>
                ))}
            </List>
        </Drawer>
    );
} 