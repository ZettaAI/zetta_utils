'use client';

import { useState, useEffect, useCallback } from 'react';
import { useSearchParams, usePathname, useRouter } from 'next/navigation';
import {
    Box,
    Paper,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    TablePagination,
    TableSortLabel,
    CircularProgress,
    Alert,
    Typography,
    Chip,
    Stack,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    TextField,
    Button,
    SelectChangeEvent,
    Switch,
    FormControlLabel,
    IconButton,
    Menu,
    Checkbox,
    ListItemText,
    Tooltip
} from '@mui/material';
import { Subtask } from '@/app/api/subtasks/route';
import { SubtaskType } from '@/app/api/subtasks/types/route';
import { formatTimestamp } from '@/lib/utils';
import ViewColumnIcon from '@mui/icons-material/ViewColumn';
import RefreshIcon from '@mui/icons-material/Refresh';

type Order = 'asc' | 'desc';

// Extend the Subtask interface to include is_complete
interface ExtendedSubtask extends Subtask {
    is_complete?: boolean;
    time_spent?: string;
    subtask_id: string; // Ensure this is defined explicitly
    time_spent_column?: any; // Add this for the new column ID
    project_id?: string; // Add project_id
}

// Add interface for time spent data
interface TimeSpentData {
    [subtaskId: string]: {
        loading: boolean;
        durationSeconds?: number;
        error?: string;
    };
}

interface UserBasic {
    user_id: string;
}

interface Column {
    id: keyof ExtendedSubtask;
    label: string;
    minWidth?: number;
    align?: 'right' | 'left' | 'center';
    format?: (value: any, subtask: ExtendedSubtask) => any;
    sortable?: boolean;
}

// Define the TimeSpentCell component
function TimeSpentCell({ subtaskId, projectId }: { subtaskId: string; projectId: string }) {
    const [timeData, setTimeData] = useState<{
        loading: boolean;
        durationSeconds?: number;
        error?: string;
    }>({ loading: true });

    // Fetch time spent data for the subtask
    useEffect(() => {
        if (!projectId || !subtaskId) return;

        const fetchTimeSpent = async () => {
            try {
                // Fetch time data from timesheets subcollection
                const response = await fetch(`/api/timesheets?projectId=${encodeURIComponent(projectId)}&subtaskId=${encodeURIComponent(subtaskId)}`);

                if (!response.ok) {
                    throw new Error('Failed to fetch time data');
                }

                const data = await response.json();
                const timeEntries = data.timeEntries || [];

                // Sum up all duration_seconds
                const totalDuration = timeEntries.reduce(
                    (sum: number, entry: { duration_seconds: number }) => sum + (entry.duration_seconds || 0),
                    0
                );

                // Update state with the result
                setTimeData({
                    loading: false,
                    durationSeconds: totalDuration
                });
            } catch (err) {
                console.error(`Error fetching time data for subtask ${subtaskId}:`, err);
                setTimeData({
                    loading: false,
                    error: 'Failed to load'
                });
            }
        };

        fetchTimeSpent();
    }, [subtaskId, projectId]);

    // Format seconds into hours, minutes and seconds
    const formatDuration = (seconds: number): string => {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;

        if (hours > 0) {
            return `${hours}h ${minutes}m ${secs}s`;
        } else if (minutes > 0) {
            return `${minutes}m ${secs}s`;
        }
        return `${secs}s`;
    };

    // Handle different states
    if (timeData.loading) {
        return <CircularProgress size={16} />;
    }

    if (timeData.error) {
        return <Typography variant="body2" color="error">Error</Typography>;
    }

    if (timeData.durationSeconds === 0) {
        return <Typography variant="body2" color="text.secondary">No time recorded</Typography>;
    }

    return (
        <Typography variant="body2">
            {formatDuration(timeData.durationSeconds || 0)}
        </Typography>
    );
}

// Original allColumns definition
const createAllColumns = (projectId: string): Column[] => [
    { id: 'subtask_id', label: 'Subtask ID', minWidth: 170 },
    { id: 'task_id', label: 'Task ID', minWidth: 120 },
    { id: 'batch_id', label: 'Batch ID', minWidth: 120 },
    {
        id: 'subtask_type',
        label: 'Type',
        minWidth: 120,
        format: (value: string) => (
            <Chip
                label={value}
                size="small"
                sx={{
                    backgroundColor: '#e8f0fe',
                    color: '#1a73e8',
                    '& .MuiChip-label': {
                        fontSize: '0.75rem'
                    }
                }}
            />
        )
    },
    {
        id: 'is_active',
        label: 'Active',
        minWidth: 80,
        align: 'center',
        format: (value: boolean) => (
            <Chip
                label={value ? 'Yes' : 'No'}
                size="small"
                color={value ? 'success' : 'default'}
                sx={{
                    '& .MuiChip-label': {
                        fontSize: '0.75rem'
                    }
                }}
            />
        )
    },
    {
        id: 'is_complete',
        label: 'Complete',
        minWidth: 80,
        align: 'center',
        format: (value: boolean) => (
            <Chip
                label={value ? 'Yes' : 'No'}
                size="small"
                color={value ? 'success' : 'default'}
                sx={{
                    '& .MuiChip-label': {
                        fontSize: '0.75rem'
                    }
                }}
            />
        )
    },
    // Add Time Spent column (using type assertion)
    {
        id: 'time_spent_column' as keyof ExtendedSubtask,
        label: 'Time Spent',
        minWidth: 120,
        sortable: false,
        format: (value: string, subtask: ExtendedSubtask) => (
            <TimeSpentCell
                subtaskId={subtask.subtask_id}
                projectId={projectId}
            />
        )
    },
    { id: 'assigned_user_id', label: 'Assigned User', minWidth: 150 },
    { id: 'active_user_id', label: 'Active User', minWidth: 150 },
    { id: 'completed_user_id', label: 'Completed User', minWidth: 150 },
    {
        id: 'last_leased_ts',
        label: 'Last Leased',
        minWidth: 150,
        format: (value: number) => formatTimestamp(value)
    },
    { id: 'priority', label: 'Priority', minWidth: 80, align: 'right' },
    { id: 'completion_status', label: 'Status', minWidth: 120 },
    {
        id: 'ng_state',
        label: 'Neuroglancer Link',
        minWidth: 120,
        format: (value: string | undefined, subtask: ExtendedSubtask) => {
            if (!value) return '-';
            try {
                const baseUrl = 'https://zetta-portal.vercel.app/?workspaces_sortValue=asc&workspaces_sortBy=name_lowercase';
                return (
                    <Button
                        variant="text"
                        size="small"
                        href={`${baseUrl}#!${encodeURIComponent(value)}`}
                        target="_blank"
                        rel="noopener noreferrer"
                    >
                        View
                    </Button>
                );
            } catch (error) {
                console.error('Error formatting Neuroglancer link:', error);
                return '-';
            }
        }
    },
];

// Default visible columns - explicitly cast to keyof ExtendedSubtask[]
const getDefaultVisibleColumns = () =>
    ['subtask_id', 'batch_id', 'subtask_type', 'is_active', 'is_complete', 'time_spent_column',
        'assigned_user_id', 'active_user_id', 'completed_user_id', 'completion_status', 'ng_state'] as (keyof ExtendedSubtask)[];

/**
 * URL Parameter System for Filter State Persistence
 * 
 * This implementation uses URL parameters to persist filter state across page reloads.
 * Each filter has a corresponding URL parameter, and the current state is reflected in the URL.
 * When the page loads, it initializes state from URL parameters.
 * When filter state changes, the URL is updated via the updateUrlWithFilters function.
 * 
 * Benefits:
 * - Users can share specific filtered views by sharing URLs
 * - Filter state persists across page reloads/refreshes
 * - Browser history works correctly with the back/forward buttons
 * 
 * Implementation notes:
 * - Default values are omitted from the URL to keep it clean
 * - A debounce (setTimeout) is applied to avoid excessive URL updates
 * - router.replace() is used instead of router.push() to avoid adding history entries for filter changes
 */

// URL parameter names for each filter
const URL_PARAMS = {
    type: 'type',
    active: 'active',
    complete: 'complete',
    assignedUser: 'assigned',
    activeUser: 'auser',
    completedUser: 'cuser',
    taskId: 'task',
    batchId: 'batch',
    status: 'status',
    lastLeasedStart: 'lstart',
    lastLeasedEnd: 'lend',
    pageNum: 'pageNum',
    rowsPerPage: 'size',
    orderBy: 'sort',
    order: 'dir',
    columns: 'cols',
};

// Helper functions for consistent type parsing/serialization
const parseUrlParams = {
    // Convert string 'true'/'false'/'null' to boolean or null
    boolean: (value: string | null): boolean | null => {
        if (value === 'true') return true;
        if (value === 'false') return false;
        if (value === 'null') return null;
        return null;
    },
    // Convert string to number, with fallback
    number: (value: string | null, fallback: number): number => {
        if (!value) return fallback;
        const parsed = parseInt(value, 10);
        return isNaN(parsed) ? fallback : parsed;
    },
    // Parse comma-separated string into array
    array: (value: string | null): string[] => {
        if (!value) return [];
        return value.split(',');
    }
};

export default function SubtasksPage() {
    const searchParams = useSearchParams();
    const pathname = usePathname();
    const router = useRouter();
    const projectId = searchParams.get('project') || '';

    // Preserve routing parameters
    const routingPage = searchParams.get('page') || 'subtasks';
    const tab = searchParams.get('tab') || 'subtasks';

    // Initialize all columns with projectId
    const allColumns = createAllColumns(projectId);
    const defaultVisibleColumns = getDefaultVisibleColumns();

    // Helper function to create URL with updated parameters
    const createQueryString = useCallback(
        (params: Record<string, string | null>) => {
            const newParams = new URLSearchParams(searchParams.toString());

            // Update or remove parameters
            Object.entries(params).forEach(([name, value]) => {
                if (value === null) {
                    newParams.delete(name);
                } else {
                    newParams.set(name, value);
                }
            });

            return newParams.toString();
        },
        [searchParams]
    );

    const [subtasks, setSubtasks] = useState<ExtendedSubtask[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Initialize pagination state from URL
    const [page, setPage] = useState(() =>
        parseUrlParams.number(searchParams.get(URL_PARAMS.pageNum), 0));

    const [rowsPerPage, setRowsPerPage] = useState(() =>
        parseUrlParams.number(searchParams.get(URL_PARAMS.rowsPerPage), 10));

    const [totalRows, setTotalRows] = useState(0);

    // Initialize sorting state from URL
    const [orderBy, setOrderBy] = useState<keyof ExtendedSubtask>(() => {
        const sortParam = searchParams.get(URL_PARAMS.orderBy);
        return sortParam ? (sortParam as keyof ExtendedSubtask) : 'subtask_id';
    });

    const [order, setOrder] = useState<Order>(() => {
        const dirParam = searchParams.get(URL_PARAMS.order);
        return (dirParam === 'desc' ? 'desc' : 'asc') as Order;
    });

    // Initialize column visibility state from URL
    const [visibleColumns, setVisibleColumns] = useState<(keyof ExtendedSubtask)[]>(() => {
        const colsParam = searchParams.get(URL_PARAMS.columns);
        if (colsParam) {
            const cols = parseUrlParams.array(colsParam) as (keyof ExtendedSubtask)[];
            return cols.length > 0 ? cols : defaultVisibleColumns;
        }
        return defaultVisibleColumns;
    });

    const [columnMenuAnchorEl, setColumnMenuAnchorEl] = useState<null | HTMLElement>(null);

    // Filter states - initialize from URL parameters
    const [subtaskTypes, setSubtaskTypes] = useState<SubtaskType[]>([]);
    const [users, setUsers] = useState<UserBasic[]>([]);

    const [selectedType, setSelectedType] = useState<string>(() =>
        searchParams.get(URL_PARAMS.type) || '');

    const [isActive, setIsActive] = useState<boolean | null>(() => {
        const activeParam = searchParams.get(URL_PARAMS.active);
        return parseUrlParams.boolean(activeParam) ?? true; // Default is true
    });

    const [isComplete, setIsComplete] = useState<boolean | null>(() =>
        parseUrlParams.boolean(searchParams.get(URL_PARAMS.complete)));

    const [assignedUser, setAssignedUser] = useState<string>(() =>
        searchParams.get(URL_PARAMS.assignedUser) || '');

    const [activeUser, setActiveUser] = useState<string>(() =>
        searchParams.get(URL_PARAMS.activeUser) || '');

    const [completedUser, setCompletedUser] = useState<string>(() =>
        searchParams.get(URL_PARAMS.completedUser) || '');

    const [taskId, setTaskId] = useState<string>(() =>
        searchParams.get(URL_PARAMS.taskId) || '');

    const [batchId, setBatchId] = useState<string>(() =>
        searchParams.get(URL_PARAMS.batchId) || '');

    const [completionStatus, setCompletionStatus] = useState<string>(() =>
        searchParams.get(URL_PARAMS.status) || '');

    const [lastLeasedStart, setLastLeasedStart] = useState<string>(() =>
        searchParams.get(URL_PARAMS.lastLeasedStart) || '');

    const [lastLeasedEnd, setLastLeasedEnd] = useState<string>(() =>
        searchParams.get(URL_PARAMS.lastLeasedEnd) || '');

    // Add a flag to track if this is the initial render
    const [isInitialRender, setIsInitialRender] = useState(true);
    useEffect(() => {
        // After the first render, set isInitialRender to false
        setIsInitialRender(false);
    }, []);

    // Function to update URL with current filter state
    const updateUrlWithFilters = useCallback(() => {
        const params: Record<string, string | null> = {};

        // Preserve routing parameters
        params['project'] = projectId;
        params['page'] = routingPage;
        params['tab'] = tab;

        // Only add parameters with non-empty values (null means remove param)
        if (selectedType) params[URL_PARAMS.type] = selectedType;
        else params[URL_PARAMS.type] = null;

        // Handle isActive toggle - explicitly include all states in URL
        if (isActive === true) params[URL_PARAMS.active] = 'true';
        else if (isActive === false) params[URL_PARAMS.active] = 'false';
        else params[URL_PARAMS.active] = 'null';

        // Handle isComplete toggle - explicitly include all states in URL
        if (isComplete === true) params[URL_PARAMS.complete] = 'true';
        else if (isComplete === false) params[URL_PARAMS.complete] = 'false';
        else params[URL_PARAMS.complete] = 'null';

        // Only add user params if they have values
        if (assignedUser) params[URL_PARAMS.assignedUser] = assignedUser;
        else params[URL_PARAMS.assignedUser] = null;

        if (activeUser) params[URL_PARAMS.activeUser] = activeUser;
        else params[URL_PARAMS.activeUser] = null;

        if (completedUser) params[URL_PARAMS.completedUser] = completedUser;
        else params[URL_PARAMS.completedUser] = null;

        // Handle other text filters
        if (taskId) params[URL_PARAMS.taskId] = taskId;
        else params[URL_PARAMS.taskId] = null;

        if (batchId) params[URL_PARAMS.batchId] = batchId;
        else params[URL_PARAMS.batchId] = null;

        if (completionStatus) params[URL_PARAMS.status] = completionStatus;
        else params[URL_PARAMS.status] = null;

        if (lastLeasedStart) params[URL_PARAMS.lastLeasedStart] = lastLeasedStart;
        else params[URL_PARAMS.lastLeasedStart] = null;

        if (lastLeasedEnd) params[URL_PARAMS.lastLeasedEnd] = lastLeasedEnd;
        else params[URL_PARAMS.lastLeasedEnd] = null;

        // Add pagination and sorting
        if (page > 0) params[URL_PARAMS.pageNum] = page.toString();
        else params[URL_PARAMS.pageNum] = null;

        if (rowsPerPage !== 10) params[URL_PARAMS.rowsPerPage] = rowsPerPage.toString();
        else params[URL_PARAMS.rowsPerPage] = null;

        if (orderBy !== 'subtask_id') params[URL_PARAMS.orderBy] = orderBy as string;
        else params[URL_PARAMS.orderBy] = null;

        if (order !== 'asc') params[URL_PARAMS.order] = order;
        else params[URL_PARAMS.order] = null;

        // Add visible columns if they differ from default
        const columnIds = visibleColumns.map(col => col.toString());
        const defaultColumnIds = defaultVisibleColumns.map(col => col.toString());
        if (JSON.stringify(columnIds.sort()) !== JSON.stringify(defaultColumnIds.sort())) {
            params[URL_PARAMS.columns] = columnIds.join(',');
        } else {
            params[URL_PARAMS.columns] = null;
        }

        // For debugging
        console.log("Updating URL with filters:", params);

        // Update URL without forcing navigation (replace rather than push)
        router.replace(`${pathname}?${createQueryString(params)}`);
    }, [
        pathname, router, createQueryString, projectId, routingPage, tab,
        selectedType, isActive, isComplete, assignedUser, activeUser, completedUser,
        taskId, batchId, completionStatus, lastLeasedStart, lastLeasedEnd,
        page, rowsPerPage, orderBy, order, visibleColumns
    ]);

    // Update the handlers for page changes
    const handleChangePage = (_: unknown, newPage: number) => {
        setPage(newPage);

        // Get existing params and update only what changed
        const params = new URLSearchParams(searchParams.toString());
        params.set(URL_PARAMS.pageNum, newPage.toString());

        // Ensure routing params are preserved
        params.set('project', projectId);
        params.set('page', routingPage);
        params.set('tab', tab);

        router.replace(`${pathname}?${params.toString()}`);
    };

    const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
        setRowsPerPage(+event.target.value);
        setPage(0);

        // Get existing params and update only what changed
        const params = new URLSearchParams(searchParams.toString());
        params.set(URL_PARAMS.rowsPerPage, event.target.value);
        params.set(URL_PARAMS.pageNum, '0');

        // Ensure routing params are preserved
        params.set('project', projectId);
        params.set('page', routingPage);
        params.set('tab', tab);

        router.replace(`${pathname}?${params.toString()}`);
    };

    const handleSort = (property: keyof ExtendedSubtask) => () => {
        const isAsc = orderBy === property && order === 'asc';
        setOrder(isAsc ? 'desc' : 'asc');
        setOrderBy(property);

        // Get existing params and update only what changed
        const params = new URLSearchParams(searchParams.toString());
        params.set(URL_PARAMS.orderBy, property as string);
        params.set(URL_PARAMS.order, isAsc ? 'desc' : 'asc');

        // Ensure routing params are preserved
        params.set('project', projectId);
        params.set('page', routingPage);
        params.set('tab', tab);

        router.replace(`${pathname}?${params.toString()}`);
    };

    // Update URL whenever filter values change
    useEffect(() => {
        // Skip URL update on initial render
        if (isInitialRender) return;

        // Use setTimeout to batch URL updates and prevent excessive renders
        const timeoutId = setTimeout(() => {
            updateUrlWithFilters();
        }, 300);

        // Clean up timeout on component unmount or when dependencies change
        return () => clearTimeout(timeoutId);
    }, [
        updateUrlWithFilters, isInitialRender,
        selectedType, isActive, isComplete, assignedUser, activeUser, completedUser,
        taskId, batchId, completionStatus, lastLeasedStart, lastLeasedEnd,
        page, rowsPerPage, orderBy, order, visibleColumns
    ]);

    const handleSelectChange = (event: SelectChangeEvent) => {
        const { name, value } = event.target;

        // Reset to first page when changing filters
        setPage(0);

        let stateUpdated = false;
        switch (name) {
            case 'subtaskType':
                setSelectedType(value);
                stateUpdated = true;
                break;
            case 'assignedUser':
                setAssignedUser(value);
                stateUpdated = true;
                break;
            case 'activeUser':
                setActiveUser(value);
                stateUpdated = true;
                break;
            case 'completedUser':
                setCompletedUser(value);
                stateUpdated = true;
                break;
            default:
                break;
        }

        // Immediately update URL after state change
        if (stateUpdated) {
            // Use setTimeout to ensure state is updated before URL changes
            setTimeout(() => {
                // Get existing params and update only what changed
                const params = new URLSearchParams(searchParams.toString());

                // Always update page number
                params.set(URL_PARAMS.pageNum, '0');

                // Update the changed parameter
                if (value === "") {
                    // If value is empty, remove the parameter
                    switch (name) {
                        case 'subtaskType':
                            params.delete(URL_PARAMS.type);
                            break;
                        case 'assignedUser':
                            params.delete(URL_PARAMS.assignedUser);
                            break;
                        case 'activeUser':
                            params.delete(URL_PARAMS.activeUser);
                            break;
                        case 'completedUser':
                            params.delete(URL_PARAMS.completedUser);
                            break;
                    }
                } else {
                    // Otherwise set the parameter
                    switch (name) {
                        case 'subtaskType':
                            params.set(URL_PARAMS.type, value);
                            break;
                        case 'assignedUser':
                            params.set(URL_PARAMS.assignedUser, value);
                            break;
                        case 'activeUser':
                            params.set(URL_PARAMS.activeUser, value);
                            break;
                        case 'completedUser':
                            params.set(URL_PARAMS.completedUser, value);
                            break;
                    }
                }

                // Ensure routing params are preserved
                params.set('project', projectId);
                params.set('page', routingPage);
                params.set('tab', tab);

                console.log(`${name} changed to "${value}" - URL params:`, params.toString());
                router.replace(`${pathname}?${params.toString()}`);
            }, 0);
        }
    };

    const handleTextChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = event.target;

        // Reset to first page when changing filters
        setPage(0);

        let stateUpdated = false;
        switch (name) {
            case 'taskId':
                setTaskId(value);
                stateUpdated = true;
                break;
            case 'batchId':
                setBatchId(value);
                stateUpdated = true;
                break;
            case 'completionStatus':
                setCompletionStatus(value);
                stateUpdated = true;
                break;
            default:
                break;
        }

        // Immediately update URL after state change
        if (stateUpdated) {
            // Use setTimeout to ensure state is updated before URL changes
            setTimeout(() => {
                // Get existing params and update only what changed
                const params = new URLSearchParams(searchParams.toString());

                // Always update page number
                params.set(URL_PARAMS.pageNum, '0');

                // Update the changed parameter
                if (value === "") {
                    // If value is empty, remove the parameter
                    switch (name) {
                        case 'taskId':
                            params.delete(URL_PARAMS.taskId);
                            break;
                        case 'batchId':
                            params.delete(URL_PARAMS.batchId);
                            break;
                        case 'completionStatus':
                            params.delete(URL_PARAMS.status);
                            break;
                    }
                } else {
                    // Otherwise set the parameter
                    switch (name) {
                        case 'taskId':
                            params.set(URL_PARAMS.taskId, value);
                            break;
                        case 'batchId':
                            params.set(URL_PARAMS.batchId, value);
                            break;
                        case 'completionStatus':
                            params.set(URL_PARAMS.status, value);
                            break;
                    }
                }

                // Ensure routing params are preserved
                params.set('project', projectId);
                params.set('page', routingPage);
                params.set('tab', tab);

                console.log(`${name} changed to "${value}" - URL params:`, params.toString());
                router.replace(`${pathname}?${params.toString()}`);
            }, 0);
        }
    };

    const handleDateChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = event.target;

        // Reset to first page when changing filters
        setPage(0);

        let stateUpdated = false;
        switch (name) {
            case 'lastLeasedStart':
                setLastLeasedStart(value ? (new Date(value).getTime() / 1000).toString() : '');
                stateUpdated = true;
                break;
            case 'lastLeasedEnd':
                setLastLeasedEnd(value ? (new Date(value).getTime() / 1000).toString() : '');
                stateUpdated = true;
                break;
            default:
                break;
        }

        // Immediately update URL after state change
        if (stateUpdated) {
            // Use setTimeout to ensure state is updated before URL changes
            setTimeout(() => {
                // Get existing params and update only what changed
                const params = new URLSearchParams(searchParams.toString());

                // Always update page number
                params.set(URL_PARAMS.pageNum, '0');

                // Update the changed parameter
                if (!value) {
                    // If value is empty, remove the parameter
                    switch (name) {
                        case 'lastLeasedStart':
                            params.delete(URL_PARAMS.lastLeasedStart);
                            break;
                        case 'lastLeasedEnd':
                            params.delete(URL_PARAMS.lastLeasedEnd);
                            break;
                    }
                } else {
                    // Otherwise set the parameter
                    const timestamp = (new Date(value).getTime() / 1000).toString();
                    switch (name) {
                        case 'lastLeasedStart':
                            params.set(URL_PARAMS.lastLeasedStart, timestamp);
                            break;
                        case 'lastLeasedEnd':
                            params.set(URL_PARAMS.lastLeasedEnd, timestamp);
                            break;
                    }
                }

                // Ensure routing params are preserved
                params.set('project', projectId);
                params.set('page', routingPage);
                params.set('tab', tab);

                console.log(`${name} changed to "${value}" - URL params:`, params.toString());
                router.replace(`${pathname}?${params.toString()}`);
            }, 0);
        }
    };

    useEffect(() => {
        // Log initial URL parameters
        console.log("Initial URL params:", {
            project: projectId,
            page: routingPage,
            tab: tab,
            isActive: isActive,
            isComplete: isComplete,
            activeParam: searchParams.get(URL_PARAMS.active),
            completeParam: searchParams.get(URL_PARAMS.complete)
        });
    }, []);

    const handleActiveChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const newValue = event.target.checked ? true : null;
        console.log("Active toggled to:", newValue);
        setIsActive(newValue);
        setPage(0);

        // Get existing params and update only what changed
        const params = new URLSearchParams(searchParams.toString());
        params.set(URL_PARAMS.active, newValue === true ? 'true' : 'null');
        params.set(URL_PARAMS.pageNum, '0');

        // Ensure routing params are preserved
        params.set('project', projectId);
        params.set('page', routingPage);
        params.set('tab', tab);

        console.log("URL with updated active state:", params.toString());
        router.replace(`${pathname}?${params.toString()}`);
    };

    const handleCompleteChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const newValue = event.target.checked ? true : null;
        console.log("Complete toggled to:", newValue);
        setIsComplete(newValue);
        setPage(0);

        // Get existing params and update only what changed
        const params = new URLSearchParams(searchParams.toString());
        params.set(URL_PARAMS.complete, newValue === true ? 'true' : 'null');
        params.set(URL_PARAMS.pageNum, '0');

        // Ensure routing params are preserved
        params.set('project', projectId);
        params.set('page', routingPage);
        params.set('tab', tab);

        console.log("URL with updated complete state:", params.toString());
        router.replace(`${pathname}?${params.toString()}`);
    };

    const handleResetFilters = () => {
        // Reset all filter states to default values
        setSelectedType('');
        setIsActive(true); // Default is true for active
        setIsComplete(null);
        setAssignedUser('');
        setActiveUser('');
        setCompletedUser('');
        setTaskId('');
        setBatchId('');
        setCompletionStatus('');
        setLastLeasedStart('');
        setLastLeasedEnd('');
        setPage(0);
        setRowsPerPage(10);
        setOrderBy('subtask_id');
        setOrder('asc');
        setError(null);

        // Clear URL parameters except for routing ones - create a completely new URLSearchParams
        const newParams = new URLSearchParams();
        newParams.set('project', projectId);
        newParams.set('page', routingPage);
        newParams.set('tab', tab);

        // Only set default values that differ from defaults
        newParams.set(URL_PARAMS.active, 'true'); // Default active state

        console.log("Reset filters - URL params:", newParams.toString());
        router.replace(`${pathname}?${newParams.toString()}`);
    };

    const handleColumnMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
        setColumnMenuAnchorEl(event.currentTarget);
    };

    const handleColumnMenuClose = () => {
        setColumnMenuAnchorEl(null);
    };

    const handleColumnToggle = (columnId: keyof ExtendedSubtask) => {
        setVisibleColumns(prev => {
            if (prev.includes(columnId)) {
                // Don't allow hiding all columns - keep at least one
                if (prev.length <= 1) {
                    return prev;
                }
                return prev.filter(id => id !== columnId);
            }
            return [...prev, columnId];
        });

        // Immediately update URL after state change
        updateUrlWithFilters();
    };

    // Filter visible columns based on state
    const columns = allColumns.filter(column => visibleColumns.includes(column.id as keyof ExtendedSubtask));

    // Fetch subtask types for filter dropdown
    useEffect(() => {
        const fetchSubtaskTypes = async () => {
            if (!projectId) return;

            try {
                const response = await fetch(`/api/subtasks/types?projectId=${projectId}`);
                if (!response.ok) {
                    throw new Error('Failed to fetch subtask types');
                }

                const data = await response.json();
                setSubtaskTypes(data.subtaskTypes || []);
            } catch (err) {
                console.error('Error fetching subtask types:', err);
            }
        };

        fetchSubtaskTypes();
    }, [projectId]);

    // Fetch users for filter dropdowns
    useEffect(() => {
        const fetchUsers = async () => {
            if (!projectId) return;

            try {
                const response = await fetch(`/api/users/list?projectId=${projectId}`);
                if (!response.ok) {
                    throw new Error('Failed to fetch users list');
                }

                const data = await response.json();
                setUsers(data.users || []);
            } catch (err) {
                console.error('Error fetching users list:', err);
            }
        };

        fetchUsers();
    }, [projectId]);

    // Add a function to handle refresh button click and reuse it in the useEffect
    const fetchSubtasks = useCallback(async () => {
        if (!projectId) return;

        try {
            setLoading(true);
            setError(null);

            // Build query parameters
            const queryParams = new URLSearchParams({
                projectId,
                page: (page + 1).toString(),
                pageSize: rowsPerPage.toString(),
                sortBy: orderBy as string,
                sortOrder: order
            });

            // Add filter parameters if they exist
            if (selectedType) queryParams.append('subtaskType', selectedType);
            if (isActive !== null) queryParams.append('isActive', isActive.toString());
            if (isComplete === true) queryParams.append('isComplete', 'true');
            // Don't add isComplete=false to query params (will filter client-side)
            if (assignedUser) queryParams.append('assignedUserId', assignedUser);
            if (activeUser) queryParams.append('activeUserId', activeUser);
            if (completedUser) queryParams.append('completedUserId', completedUser);
            if (taskId) queryParams.append('taskId', taskId);
            if (batchId) queryParams.append('batchId', batchId);
            if (completionStatus) queryParams.append('completionStatus', completionStatus);
            if (lastLeasedStart) queryParams.append('lastLeasedTsStart', lastLeasedStart);
            if (lastLeasedEnd) queryParams.append('lastLeasedTsEnd', lastLeasedEnd);

            console.log("Fetching subtasks with params:", queryParams.toString());
            const response = await fetch(`/api/subtasks?${queryParams}`);
            if (!response.ok) {
                throw new Error('Failed to fetch subtasks');
            }

            const data = await response.json();

            // Add is_complete property based on completion_status
            let processedSubtasks = (data.subtasks || []).map((subtask: ExtendedSubtask) => ({
                ...subtask,
                is_complete: subtask.completion_status !== ''
            }));

            // Client-side filtering for isComplete=false (not completed)
            if (isComplete === false) {
                processedSubtasks = processedSubtasks.filter((subtask: ExtendedSubtask) => subtask.completion_status === '');

                // Adjust total count for pagination
                setTotalRows(processedSubtasks.length);
            } else {
                setTotalRows(data.total || 0);
            }

            setSubtasks(processedSubtasks);
        } catch (err) {
            console.error('Error fetching subtasks:', err);
            setError('Failed to load subtasks');
        } finally {
            setLoading(false);
        }
    }, [projectId, page, rowsPerPage, orderBy, order, selectedType, isActive, isComplete, assignedUser, activeUser, completedUser, taskId, batchId, completionStatus, lastLeasedStart, lastLeasedEnd]);

    // Fetch subtasks when dependencies change
    useEffect(() => {
        fetchSubtasks();
    }, [fetchSubtasks]);

    const handleRefresh = () => {
        // Log the refresh action
        console.log("Manually refreshing subtask data with current filters");

        // Call the fetch function directly
        fetchSubtasks();
    };

    return (
        <Paper sx={{ width: '100%', overflow: 'hidden' }}>
            <Box sx={{ p: 2, borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="h6">Filters</Typography>
                    <Box sx={{ display: 'flex', gap: 1 }}>
                        <Tooltip title="Refresh data">
                            <IconButton onClick={handleRefresh} disabled={loading}>
                                <RefreshIcon />
                            </IconButton>
                        </Tooltip>
                        <Tooltip title="Configure columns">
                            <IconButton onClick={handleColumnMenuOpen}>
                                <ViewColumnIcon />
                            </IconButton>
                        </Tooltip>
                    </Box>
                    <Menu
                        anchorEl={columnMenuAnchorEl}
                        open={Boolean(columnMenuAnchorEl)}
                        onClose={handleColumnMenuClose}
                    >
                        {allColumns.map((column) => (
                            <MenuItem
                                key={column.id.toString()}
                                onClick={() => handleColumnToggle(column.id as keyof ExtendedSubtask)}
                                dense
                            >
                                <Checkbox
                                    checked={visibleColumns.includes(column.id as keyof ExtendedSubtask)}
                                    disabled={visibleColumns.length === 1 && visibleColumns.includes(column.id as keyof ExtendedSubtask)}
                                />
                                <ListItemText primary={column.label} />
                            </MenuItem>
                        ))}
                    </Menu>
                </Box>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                    <Box sx={{ flex: '1 1 200px', minWidth: '200px', maxWidth: '300px' }}>
                        <FormControl fullWidth size="small">
                            <InputLabel id="subtask-type-label">Subtask Type</InputLabel>
                            <Select
                                labelId="subtask-type-label"
                                name="subtaskType"
                                value={selectedType}
                                label="Subtask Type"
                                onChange={handleSelectChange}
                            >
                                <MenuItem value="">
                                    <em>All</em>
                                </MenuItem>
                                {subtaskTypes.map((type) => (
                                    <MenuItem key={type.subtask_type} value={type.subtask_type}>
                                        {type.subtask_type}
                                    </MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                    </Box>
                    <Box sx={{ flex: '1 1 200px', minWidth: '200px', maxWidth: '300px' }}>
                        <TextField
                            fullWidth
                            size="small"
                            label="Task ID"
                            name="taskId"
                            value={taskId}
                            onChange={handleTextChange}
                            placeholder="Filter by task ID"
                        />
                    </Box>
                    <Box sx={{ flex: '1 1 200px', minWidth: '200px', maxWidth: '300px' }}>
                        <TextField
                            fullWidth
                            size="small"
                            label="Batch ID"
                            name="batchId"
                            value={batchId}
                            onChange={handleTextChange}
                            placeholder="Filter by batch ID"
                        />
                    </Box>
                    <Box sx={{ flex: '1 1 200px', minWidth: '200px', maxWidth: '300px' }}>
                        <TextField
                            fullWidth
                            size="small"
                            label="Status"
                            name="completionStatus"
                            value={completionStatus}
                            onChange={handleTextChange}
                            placeholder="Filter by status"
                        />
                    </Box>
                    <Box sx={{ flex: '1 1 200px', minWidth: '200px', maxWidth: '300px', display: 'flex', alignItems: 'center' }}>
                        <FormControlLabel
                            control={
                                <Switch
                                    checked={isActive === true}
                                    onChange={handleActiveChange}
                                    name="isActive"
                                />
                            }
                            label="Active only"
                        />
                    </Box>
                    <Box sx={{ flex: '1 1 200px', minWidth: '200px', maxWidth: '300px', display: 'flex', alignItems: 'center' }}>
                        <FormControlLabel
                            control={
                                <Switch
                                    checked={isComplete === true}
                                    onChange={handleCompleteChange}
                                    name="isComplete"
                                />
                            }
                            label="Complete only"
                        />
                    </Box>
                    <Box sx={{ flex: '1 1 200px', minWidth: '200px', maxWidth: '300px' }}>
                        <FormControl fullWidth size="small">
                            <InputLabel id="assigned-user-label">Assigned User</InputLabel>
                            <Select
                                labelId="assigned-user-label"
                                name="assignedUser"
                                value={assignedUser}
                                label="Assigned User"
                                onChange={handleSelectChange}
                            >
                                <MenuItem value="">
                                    <em>All</em>
                                </MenuItem>
                                {users.map((user) => (
                                    <MenuItem key={user.user_id} value={user.user_id}>
                                        {user.user_id}
                                    </MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                    </Box>
                    <Box sx={{ flex: '1 1 200px', minWidth: '200px', maxWidth: '300px' }}>
                        <FormControl fullWidth size="small">
                            <InputLabel id="active-user-label">Active User</InputLabel>
                            <Select
                                labelId="active-user-label"
                                name="activeUser"
                                value={activeUser}
                                label="Active User"
                                onChange={handleSelectChange}
                            >
                                <MenuItem value="">
                                    <em>All</em>
                                </MenuItem>
                                {users.map((user) => (
                                    <MenuItem key={user.user_id} value={user.user_id}>
                                        {user.user_id}
                                    </MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                    </Box>
                    <Box sx={{ flex: '1 1 200px', minWidth: '200px', maxWidth: '300px' }}>
                        <FormControl fullWidth size="small">
                            <InputLabel id="completed-user-label">Completed User</InputLabel>
                            <Select
                                labelId="completed-user-label"
                                name="completedUser"
                                value={completedUser}
                                label="Completed User"
                                onChange={handleSelectChange}
                            >
                                <MenuItem value="">
                                    <em>All</em>
                                </MenuItem>
                                {users.map((user) => (
                                    <MenuItem key={user.user_id} value={user.user_id}>
                                        {user.user_id}
                                    </MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                    </Box>
                    <Box sx={{ flex: '1 1 200px', minWidth: '200px', maxWidth: '300px' }}>
                        <TextField
                            fullWidth
                            size="small"
                            label="Leased After"
                            type="datetime-local"
                            name="lastLeasedStart"
                            value={lastLeasedStart ? new Date(parseFloat(lastLeasedStart) * 1000).toISOString().slice(0, 16) : ''}
                            onChange={handleDateChange}
                            InputLabelProps={{ shrink: true }}
                        />
                    </Box>
                    <Box sx={{ flex: '1 1 200px', minWidth: '200px', maxWidth: '300px' }}>
                        <TextField
                            fullWidth
                            size="small"
                            label="Leased Before"
                            type="datetime-local"
                            name="lastLeasedEnd"
                            value={lastLeasedEnd ? new Date(parseFloat(lastLeasedEnd) * 1000).toISOString().slice(0, 16) : ''}
                            onChange={handleDateChange}
                            InputLabelProps={{ shrink: true }}
                        />
                    </Box>
                    <Box sx={{ flex: '100%', mt: 2, display: 'flex', gap: 2, alignItems: 'center' }}>
                        <Button variant="outlined" onClick={handleResetFilters}>
                            Reset Filters
                        </Button>
                        <Button
                            variant="outlined"
                            onClick={handleRefresh}
                            disabled={loading}
                            startIcon={<RefreshIcon />}
                        >
                            Refresh
                        </Button>
                        {error && (
                            <Alert
                                severity="error"
                                sx={{ flexGrow: 1 }}
                                onClose={() => setError(null)}
                            >
                                {error}
                            </Alert>
                        )}
                    </Box>
                </Box>
            </Box>

            {loading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                    <CircularProgress />
                </Box>
            ) : (
                <>
                    <TableContainer sx={{ maxHeight: 'calc(100vh - 350px)' }}>
                        <Table stickyHeader>
                            <TableHead>
                                <TableRow>
                                    {columns.map((column) => (
                                        <TableCell
                                            key={column.id}
                                            align={column.align}
                                            style={{ minWidth: column.minWidth }}
                                        >
                                            {column.sortable === false ? (
                                                column.label
                                            ) : (
                                                <TableSortLabel
                                                    active={orderBy === column.id}
                                                    direction={orderBy === column.id ? order : 'asc'}
                                                    onClick={handleSort(column.id as keyof ExtendedSubtask)}
                                                >
                                                    {column.label}
                                                </TableSortLabel>
                                            )}
                                        </TableCell>
                                    ))}
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {subtasks.length === 0 ? (
                                    <TableRow>
                                        <TableCell colSpan={columns.length} align="center">
                                            <Typography variant="body1" color="text.secondary" sx={{ py: 2 }}>
                                                {error ? 'Error loading subtasks. Try modifying filters or refreshing the page.' : 'No subtasks found'}
                                            </Typography>
                                        </TableCell>
                                    </TableRow>
                                ) : (
                                    subtasks.map((subtask) => (
                                        <TableRow hover key={subtask.subtask_id}>
                                            {columns.map((column) => {
                                                const value = subtask[column.id];
                                                return (
                                                    <TableCell key={column.id} align={column.align}>
                                                        {column.format ? column.format(value, subtask) : value}
                                                    </TableCell>
                                                );
                                            })}
                                        </TableRow>
                                    ))
                                )}
                            </TableBody>
                        </Table>
                    </TableContainer>
                    <TablePagination
                        rowsPerPageOptions={[5, 10, 25, 50]}
                        component="div"
                        count={totalRows}
                        rowsPerPage={rowsPerPage}
                        page={page}
                        onPageChange={handleChangePage}
                        onRowsPerPageChange={handleChangeRowsPerPage}
                    />
                </>
            )}
        </Paper>
    );
} 