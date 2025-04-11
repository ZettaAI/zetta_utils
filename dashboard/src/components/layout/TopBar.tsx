import { AppBar, Toolbar, IconButton, Typography, Box } from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import ProjectSelector from './ProjectSelector';
import { useDashboard } from './DashboardLayout';

interface TopBarProps {
    onMenuClick: () => void;
}

export default function TopBar({ onMenuClick }: TopBarProps) {
    const { currentPage } = useDashboard();

    const formatPageTitle = (page: string) => {
        return page.charAt(0).toUpperCase() + page.slice(1).replace(/-/g, ' ');
    };

    return (
        <AppBar
            position="fixed"
            sx={{
                zIndex: (theme) => theme.zIndex.drawer + 1,
                bgcolor: 'white',
                color: 'text.primary',
                boxShadow: 1
            }}
        >
            <Toolbar>
                <IconButton
                    edge="start"
                    color="inherit"
                    aria-label="menu"
                    onClick={onMenuClick}
                    sx={{ mr: 2 }}
                >
                    <MenuIcon />
                </IconButton>

                <Typography
                    variant="h6"
                    sx={{
                        flexGrow: 1,
                        fontWeight: 500,
                        color: (theme) => theme.palette.grey[800]
                    }}
                >
                    {formatPageTitle(currentPage)}
                </Typography>

                <Box sx={{ ml: 2 }}>
                    <ProjectSelector />
                </Box>
            </Toolbar>
        </AppBar>
    );
} 