'use client';

import { useState, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
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

type Order = 'asc' | 'desc';

// Extend the Subtask interface to include is_complete
interface ExtendedSubtask extends Subtask {
    is_complete?: boolean;
    time_spent?: string;
    subtask_id: string; // Ensure this is defined explicitly
    time_spent_column?: any; // Add this for the new column ID
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

const allColumns: Column[] = [
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
                projectId={new URLSearchParams(window.location.search).get('project') || ''}
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
const defaultVisibleColumns = allColumns
    .filter(col => !['priority', 'task_id', 'last_leased_ts'].includes(col.id as string))
    .map(col => col.id) as (keyof ExtendedSubtask)[];

export default function SubtasksPage() {
    const searchParams = useSearchParams();
    const projectId = searchParams.get('project') || '';

    const [subtasks, setSubtasks] = useState<ExtendedSubtask[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [page, setPage] = useState(0);
    const [rowsPerPage, setRowsPerPage] = useState(10);
    const [totalRows, setTotalRows] = useState(0);
    const [orderBy, setOrderBy] = useState<keyof ExtendedSubtask>('subtask_id');
    const [order, setOrder] = useState<Order>('asc');

    // Column visibility state
    const [visibleColumns, setVisibleColumns] = useState<(keyof ExtendedSubtask)[]>(defaultVisibleColumns);
    const [columnMenuAnchorEl, setColumnMenuAnchorEl] = useState<null | HTMLElement>(null);

    // Filter states
    const [subtaskTypes, setSubtaskTypes] = useState<SubtaskType[]>([]);
    const [users, setUsers] = useState<UserBasic[]>([]);
    const [selectedType, setSelectedType] = useState<string>('');
    const [isActive, setIsActive] = useState<boolean | null>(true);
    const [isComplete, setIsComplete] = useState<boolean | null>(null);
    const [assignedUser, setAssignedUser] = useState<string>('');
    const [activeUser, setActiveUser] = useState<string>('');
    const [completedUser, setCompletedUser] = useState<string>('');
    const [taskId, setTaskId] = useState<string>('');
    const [batchId, setBatchId] = useState<string>('');
    const [completionStatus, setCompletionStatus] = useState<string>('');
    const [lastLeasedStart, setLastLeasedStart] = useState<string>('');
    const [lastLeasedEnd, setLastLeasedEnd] = useState<string>('');

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

    // Fetch subtasks with pagination, sorting and filtering
    useEffect(() => {
        const fetchSubtasks = async () => {
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
        };

        fetchSubtasks();
    }, [projectId, page, rowsPerPage, orderBy, order, selectedType, isActive, isComplete, assignedUser, activeUser, completedUser, taskId, batchId, completionStatus, lastLeasedStart, lastLeasedEnd]);

    const handleChangePage = (_: unknown, newPage: number) => {
        setPage(newPage);
    };

    const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
        setRowsPerPage(+event.target.value);
        setPage(0);
    };

    const handleSort = (property: keyof ExtendedSubtask) => () => {
        const isAsc = orderBy === property && order === 'asc';
        setOrder(isAsc ? 'desc' : 'asc');
        setOrderBy(property);
    };

    const handleSelectChange = (event: SelectChangeEvent) => {
        const { name, value } = event.target;

        // Reset to first page when changing filters
        setPage(0);

        switch (name) {
            case 'subtaskType':
                setSelectedType(value);
                break;
            case 'assignedUser':
                setAssignedUser(value);
                break;
            case 'activeUser':
                setActiveUser(value);
                break;
            case 'completedUser':
                setCompletedUser(value);
                break;
            default:
                break;
        }
    };

    const handleTextChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = event.target;

        // Reset to first page when changing filters
        setPage(0);

        switch (name) {
            case 'taskId':
                setTaskId(value);
                break;
            case 'batchId':
                setBatchId(value);
                break;
            case 'completionStatus':
                setCompletionStatus(value);
                break;
            default:
                break;
        }
    };

    const handleDateChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = event.target;

        // Reset to first page when changing filters
        setPage(0);

        switch (name) {
            case 'lastLeasedStart':
                setLastLeasedStart(value ? (new Date(value).getTime() / 1000).toString() : '');
                break;
            case 'lastLeasedEnd':
                setLastLeasedEnd(value ? (new Date(value).getTime() / 1000).toString() : '');
                break;
            default:
                break;
        }
    };

    const handleActiveChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setIsActive(event.target.checked ? true : null);
        setPage(0);
    };

    const handleCompleteChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setIsComplete(event.target.checked ? true : null);
        setPage(0);
    };

    const handleResetFilters = () => {
        setSelectedType('');
        setIsActive(null);
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
        setError(null);
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
    };

    return (
        <Paper sx={{ width: '100%', overflow: 'hidden' }}>
            <Box sx={{ p: 2, borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="h6">Filters</Typography>
                    <Tooltip title="Configure columns">
                        <IconButton onClick={handleColumnMenuOpen}>
                            <ViewColumnIcon />
                        </IconButton>
                    </Tooltip>
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