import { ReactNode, useState, createContext, useContext, useEffect } from 'react';
import { Box } from '@mui/material';
import { useRouter, useSearchParams } from 'next/navigation';
import TopBar from './TopBar';
import SideNav from './SideNav';

interface DashboardContextType {
    currentProject: string | null;
    setCurrentProject: (project: string) => void;
    currentPage: string;
    setCurrentPage: (page: string) => void;
}

export const DashboardContext = createContext<DashboardContextType>({
    currentProject: null,
    setCurrentProject: () => { },
    currentPage: 'progress',
    setCurrentPage: () => { },
});

export function useDashboard() {
    return useContext(DashboardContext);
}

interface DashboardLayoutProps {
    children: ReactNode;
}

export default function DashboardLayout({ children }: DashboardLayoutProps) {
    const router = useRouter();
    const searchParams = useSearchParams();
    const [currentProject, setCurrentProject] = useState<string | null>(null);
    const [currentPage, setCurrentPage] = useState('progress');
    const [sideNavOpen, setSideNavOpen] = useState(false);

    // Read initial values from URL
    useEffect(() => {
        const pageFromUrl = searchParams.get('page');
        const projectFromUrl = searchParams.get('project');

        if (pageFromUrl) {
            setCurrentPage(pageFromUrl);
        }
        if (projectFromUrl) {
            setCurrentProject(projectFromUrl);
        }
    }, [searchParams]);

    // Update URL when values change
    const handleSetCurrentProject = (project: string) => {
        setCurrentProject(project);
        const params = new URLSearchParams(searchParams.toString());
        params.set('project', project);
        router.push(`?${params.toString()}`);
    };

    const handleSetCurrentPage = (page: string) => {
        setCurrentPage(page);
        const params = new URLSearchParams(searchParams.toString());
        params.set('page', page);
        router.push(`?${params.toString()}`);
    };

    return (
        <DashboardContext.Provider value={{
            currentProject,
            setCurrentProject: handleSetCurrentProject,
            currentPage,
            setCurrentPage: handleSetCurrentPage,
        }}>
            <Box sx={{ display: 'flex', minHeight: '100vh' }}>
                <TopBar onMenuClick={() => setSideNavOpen(true)} />
                <SideNav
                    open={sideNavOpen}
                    onClose={() => setSideNavOpen(false)}
                />
                <Box
                    component="main"
                    sx={{
                        flexGrow: 1,
                        pt: '64px', // Height of TopBar
                        bgcolor: 'background.default',
                    }}
                >
                    {children}
                </Box>
            </Box>
        </DashboardContext.Provider>
    );
} 