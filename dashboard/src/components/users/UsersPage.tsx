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
    Button,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    TextField,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    Checkbox,
    ListItemText,
    OutlinedInput,
    FormHelperText,
    IconButton,
    Tooltip
} from '@mui/material';
import { User } from '@/app/api/users/route';
import { SubtaskType } from '@/app/api/subtasks/types/route';
import AddIcon from '@mui/icons-material/Add';
import EditIcon from '@mui/icons-material/Edit';
import DeleteIcon from '@mui/icons-material/Delete';
import { SelectChangeEvent } from '@mui/material';

type Order = 'asc' | 'desc';

interface Column {
    id: keyof User;
    label: string;
    minWidth?: number;
    align?: 'right' | 'left' | 'center';
    format?: (value: any) => any;
}

const columns: Column[] = [
    { id: 'user_id', label: 'Email', minWidth: 200 },
    {
        id: 'qualified_subtask_types',
        label: 'Qualified Subtask Types',
        minWidth: 300,
        format: (value: string[]) => (
            <Stack direction="row" spacing={1} flexWrap="wrap" gap={1}>
                {value?.map((type) => (
                    <Chip
                        key={type}
                        label={type.replace('segmentation_', '')}
                        size="small"
                        sx={{
                            backgroundColor: '#e8f0fe',
                            color: '#1a73e8',
                            '& .MuiChip-label': {
                                fontSize: '0.75rem'
                            }
                        }}
                    />
                )) || 'None'}
            </Stack>
        )
    },
    {
        id: 'hourly_rate',
        label: 'Rate ($/hr)',
        minWidth: 120,
        align: 'right',
        format: (value: number) => value?.toFixed(2) || '0.00'
    },
    {
        id: 'active_subtask',
        label: 'Active Task',
        minWidth: 170,
        format: (value: string) => value || 'None'
    }
];

// Interface for new user form
interface NewUserForm {
    user_id: string;
    hourly_rate: string;
    qualified_subtask_types: string[];
}

// Interface for edit user form
interface EditUserForm {
    hourly_rate: string;
    qualified_subtask_types: string[];
}

export default function UsersPage() {
    const searchParams = useSearchParams();
    const projectId = searchParams.get('project') || '';

    const [users, setUsers] = useState<User[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [page, setPage] = useState(0);
    const [rowsPerPage, setRowsPerPage] = useState(10);
    const [totalRows, setTotalRows] = useState(0);
    const [orderBy, setOrderBy] = useState<keyof User>('user_id');
    const [order, setOrder] = useState<Order>('asc');

    // Add User Dialog states
    const [openAddUserDialog, setOpenAddUserDialog] = useState(false);
    const [subtaskTypes, setSubtaskTypes] = useState<SubtaskType[]>([]);
    const [newUser, setNewUser] = useState<NewUserForm>({
        user_id: '',
        hourly_rate: '0',
        qualified_subtask_types: []
    });
    const [formErrors, setFormErrors] = useState<{ [key: string]: string }>({});
    const [addUserLoading, setAddUserLoading] = useState(false);
    const [addUserError, setAddUserError] = useState<string | null>(null);

    // Edit User Dialog states
    const [openEditUserDialog, setOpenEditUserDialog] = useState(false);
    const [editUser, setEditUser] = useState<EditUserForm>({
        hourly_rate: '0',
        qualified_subtask_types: []
    });
    const [editUserId, setEditUserId] = useState<string>('');
    const [editFormErrors, setEditFormErrors] = useState<{ [key: string]: string }>({});
    const [editUserLoading, setEditUserLoading] = useState(false);
    const [editUserError, setEditUserError] = useState<string | null>(null);

    // Delete User Dialog states
    const [openDeleteDialog, setOpenDeleteDialog] = useState(false);
    const [deleteUserId, setDeleteUserId] = useState<string>('');
    const [deleteUserLoading, setDeleteUserLoading] = useState(false);
    const [deleteUserError, setDeleteUserError] = useState<string | null>(null);

    // Fetch subtask types for the add user form
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

    const fetchUsers = async () => {
        if (!projectId) return;

        try {
            setLoading(true);
            setError(null);

            const queryParams = new URLSearchParams({
                projectId,
                page: (page + 1).toString(),
                pageSize: rowsPerPage.toString(),
                sortBy: orderBy,
                sortOrder: order
            });

            const response = await fetch(`/api/users?${queryParams}`);
            if (!response.ok) {
                throw new Error('Failed to fetch users');
            }

            const data = await response.json();
            setUsers(data.users);
            setTotalRows(data.total);
        } catch (err) {
            console.error('Error fetching users:', err);
            setError('Failed to load users');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchUsers();
    }, [projectId, page, rowsPerPage, orderBy, order]);

    const handleChangePage = (_: unknown, newPage: number) => {
        setPage(newPage);
    };

    const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
        setRowsPerPage(+event.target.value);
        setPage(0);
    };

    const handleSort = (property: keyof User) => () => {
        const isAsc = orderBy === property && order === 'asc';
        setOrder(isAsc ? 'desc' : 'asc');
        setOrderBy(property);
    };

    // Add User Dialog handlers
    const handleOpenAddUserDialog = () => {
        setOpenAddUserDialog(true);
        setNewUser({
            user_id: '',
            hourly_rate: '0',
            qualified_subtask_types: []
        });
        setFormErrors({});
        setAddUserError(null);
    };

    const handleCloseAddUserDialog = () => {
        setOpenAddUserDialog(false);
    };

    const handleNewUserChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = event.target;
        setNewUser(prev => ({
            ...prev,
            [name]: value
        }));
        // Clear error for this field if it exists
        if (formErrors[name]) {
            setFormErrors(prev => {
                const newErrors = { ...prev };
                delete newErrors[name];
                return newErrors;
            });
        }
    };

    const handleSubtaskTypesChange = (event: SelectChangeEvent<string[]>) => {
        const value = event.target.value as string[];
        setNewUser(prev => ({
            ...prev,
            qualified_subtask_types: value
        }));
        // Clear error for this field if it exists
        if (formErrors.qualified_subtask_types) {
            setFormErrors(prev => {
                const newErrors = { ...prev };
                delete newErrors.qualified_subtask_types;
                return newErrors;
            });
        }
    };

    // Edit User Dialog handlers
    const handleOpenEditUserDialog = (user: User) => {
        setOpenEditUserDialog(true);
        setEditUserId(user.user_id);
        setEditUser({
            hourly_rate: user.hourly_rate.toString(),
            qualified_subtask_types: user.qualified_subtask_types || []
        });
        setEditFormErrors({});
        setEditUserError(null);
    };

    const handleCloseEditUserDialog = () => {
        setOpenEditUserDialog(false);
    };

    const handleEditUserChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = event.target;
        setEditUser(prev => ({
            ...prev,
            [name]: value
        }));
        // Clear error for this field if it exists
        if (editFormErrors[name]) {
            setEditFormErrors(prev => {
                const newErrors = { ...prev };
                delete newErrors[name];
                return newErrors;
            });
        }
    };

    const handleEditSubtaskTypesChange = (event: SelectChangeEvent<string[]>) => {
        const value = event.target.value as string[];
        setEditUser(prev => ({
            ...prev,
            qualified_subtask_types: value
        }));
        // Clear error for this field if it exists
        if (editFormErrors.qualified_subtask_types) {
            setEditFormErrors(prev => {
                const newErrors = { ...prev };
                delete newErrors.qualified_subtask_types;
                return newErrors;
            });
        }
    };

    // Delete User Dialog handlers
    const handleOpenDeleteDialog = (userId: string) => {
        setOpenDeleteDialog(true);
        setDeleteUserId(userId);
        setDeleteUserError(null);
    };

    const handleCloseDeleteDialog = () => {
        setOpenDeleteDialog(false);
    };

    const validateForm = () => {
        const errors: { [key: string]: string } = {};

        if (!newUser.user_id) {
            errors.user_id = 'Email is required';
        } else if (!/\S+@\S+\.\S+/.test(newUser.user_id)) {
            errors.user_id = 'Must be a valid email address';
        }

        if (!newUser.hourly_rate) {
            errors.hourly_rate = 'Hourly rate is required';
        } else if (isNaN(parseFloat(newUser.hourly_rate)) || parseFloat(newUser.hourly_rate) < 0) {
            errors.hourly_rate = 'Must be a valid positive number';
        }

        setFormErrors(errors);
        return Object.keys(errors).length === 0;
    };

    const validateEditForm = () => {
        const errors: { [key: string]: string } = {};

        if (!editUser.hourly_rate) {
            errors.hourly_rate = 'Hourly rate is required';
        } else if (isNaN(parseFloat(editUser.hourly_rate)) || parseFloat(editUser.hourly_rate) < 0) {
            errors.hourly_rate = 'Must be a valid positive number';
        }

        setEditFormErrors(errors);
        return Object.keys(errors).length === 0;
    };

    const handleAddUser = async () => {
        if (!validateForm()) return;

        setAddUserLoading(true);
        setAddUserError(null);

        try {
            const userData = {
                user_id: newUser.user_id,
                hourly_rate: parseFloat(newUser.hourly_rate),
                qualified_subtask_types: newUser.qualified_subtask_types,
                active_subtask: ""  // New users don't have an active subtask
            };

            const response = await fetch(`/api/users/add`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    projectId,
                    userData
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to add user');
            }

            // Refresh the user list
            await fetchUsers();

            handleCloseAddUserDialog();
        } catch (err: any) {
            console.error('Error adding user:', err);
            setAddUserError(err.message || 'Failed to add user');
        } finally {
            setAddUserLoading(false);
        }
    };

    const handleUpdateUser = async () => {
        if (!validateEditForm()) return;

        setEditUserLoading(true);
        setEditUserError(null);

        try {
            const userData = {
                hourly_rate: parseFloat(editUser.hourly_rate),
                qualified_subtask_types: editUser.qualified_subtask_types,
            };

            const response = await fetch(`/api/users/update`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    projectId,
                    userId: editUserId,
                    userData
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to update user');
            }

            // Refresh the user list
            await fetchUsers();

            handleCloseEditUserDialog();
        } catch (err: any) {
            console.error('Error updating user:', err);
            setEditUserError(err.message || 'Failed to update user');
        } finally {
            setEditUserLoading(false);
        }
    };

    const handleDeleteUser = async () => {
        setDeleteUserLoading(true);
        setDeleteUserError(null);

        try {
            const response = await fetch(`/api/users/delete`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    projectId,
                    userId: deleteUserId
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to delete user');
            }

            // Refresh the user list
            await fetchUsers();

            handleCloseDeleteDialog();
        } catch (err: any) {
            console.error('Error deleting user:', err);
            setDeleteUserError(err.message || 'Failed to delete user');
        } finally {
            setDeleteUserLoading(false);
        }
    };

    if (error) {
        return <Alert severity="error">{error}</Alert>;
    }

    if (loading) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                <CircularProgress />
            </Box>
        );
    }

    return (
        <Paper sx={{ width: '100%', overflow: 'hidden' }}>
            <Box sx={{ p: 2, display: 'flex', justifyContent: 'flex-end' }}>
                <Button
                    variant="contained"
                    startIcon={<AddIcon />}
                    onClick={handleOpenAddUserDialog}
                >
                    Add User
                </Button>
            </Box>

            <TableContainer sx={{ maxHeight: 'calc(100vh - 250px)' }}>
                <Table stickyHeader>
                    <TableHead>
                        <TableRow>
                            {columns.map((column) => (
                                <TableCell
                                    key={column.id}
                                    align={column.align}
                                    style={{ minWidth: column.minWidth }}
                                >
                                    <TableSortLabel
                                        active={orderBy === column.id}
                                        direction={orderBy === column.id ? order : 'asc'}
                                        onClick={handleSort(column.id)}
                                    >
                                        {column.label}
                                    </TableSortLabel>
                                </TableCell>
                            ))}
                            <TableCell align="center" style={{ minWidth: 100 }}>
                                Actions
                            </TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {users.length === 0 ? (
                            <TableRow>
                                <TableCell colSpan={columns.length + 1} align="center">
                                    <Typography variant="body1" color="text.secondary" sx={{ py: 2 }}>
                                        No users found
                                    </Typography>
                                </TableCell>
                            </TableRow>
                        ) : (
                            users.map((user) => (
                                <TableRow hover key={user.user_id}>
                                    {columns.map((column) => {
                                        const value = user[column.id];
                                        return (
                                            <TableCell key={column.id} align={column.align}>
                                                {column.format ? column.format(value) : value}
                                            </TableCell>
                                        );
                                    })}
                                    <TableCell align="center">
                                        <Stack direction="row" spacing={1} justifyContent="center">
                                            <Tooltip title="Edit user">
                                                <IconButton
                                                    size="small"
                                                    color="primary"
                                                    onClick={() => handleOpenEditUserDialog(user)}
                                                >
                                                    <EditIcon fontSize="small" />
                                                </IconButton>
                                            </Tooltip>
                                            <Tooltip title="Delete user">
                                                <IconButton
                                                    size="small"
                                                    color="error"
                                                    onClick={() => handleOpenDeleteDialog(user.user_id)}
                                                >
                                                    <DeleteIcon fontSize="small" />
                                                </IconButton>
                                            </Tooltip>
                                        </Stack>
                                    </TableCell>
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

            {/* Add User Dialog */}
            <Dialog open={openAddUserDialog} onClose={handleCloseAddUserDialog} fullWidth maxWidth="sm">
                <DialogTitle>Add New User</DialogTitle>
                <DialogContent>
                    {addUserError && (
                        <Alert severity="error" sx={{ mb: 2 }}>
                            {addUserError}
                        </Alert>
                    )}
                    <Box sx={{ mt: 2, display: 'flex', flexDirection: 'column', gap: 2 }}>
                        <TextField
                            label="Email"
                            name="user_id"
                            value={newUser.user_id}
                            onChange={handleNewUserChange}
                            fullWidth
                            error={!!formErrors.user_id}
                            helperText={formErrors.user_id}
                            disabled={addUserLoading}
                        />

                        <TextField
                            label="Hourly Rate ($)"
                            name="hourly_rate"
                            type="number"
                            value={newUser.hourly_rate}
                            onChange={handleNewUserChange}
                            fullWidth
                            error={!!formErrors.hourly_rate}
                            helperText={formErrors.hourly_rate}
                            disabled={addUserLoading}
                            inputProps={{ step: "0.01", min: "0" }}
                        />

                        <FormControl fullWidth disabled={addUserLoading}>
                            <InputLabel id="qualified-subtask-types-label">
                                Qualified Subtask Types
                            </InputLabel>
                            <Select
                                labelId="qualified-subtask-types-label"
                                multiple
                                value={newUser.qualified_subtask_types}
                                onChange={handleSubtaskTypesChange}
                                input={<OutlinedInput label="Qualified Subtask Types" />}
                                renderValue={(selected) => (
                                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                                        {selected.map((value) => (
                                            <Chip key={value} label={value} />
                                        ))}
                                    </Box>
                                )}
                            >
                                {subtaskTypes.map((type) => (
                                    <MenuItem key={type.subtask_type} value={type.subtask_type}>
                                        <Checkbox checked={newUser.qualified_subtask_types.indexOf(type.subtask_type) > -1} />
                                        <ListItemText primary={type.subtask_type} />
                                    </MenuItem>
                                ))}
                            </Select>
                            <FormHelperText>Select all subtask types this user is qualified to work on</FormHelperText>
                        </FormControl>
                    </Box>
                </DialogContent>
                <DialogActions>
                    <Button onClick={handleCloseAddUserDialog} disabled={addUserLoading}>
                        Cancel
                    </Button>
                    <Button
                        onClick={handleAddUser}
                        variant="contained"
                        disabled={addUserLoading}
                        startIcon={addUserLoading ? <CircularProgress size={20} /> : null}
                    >
                        {addUserLoading ? 'Adding...' : 'Add User'}
                    </Button>
                </DialogActions>
            </Dialog>

            {/* Edit User Dialog */}
            <Dialog open={openEditUserDialog} onClose={handleCloseEditUserDialog} fullWidth maxWidth="sm">
                <DialogTitle>Edit User: {editUserId}</DialogTitle>
                <DialogContent>
                    {editUserError && (
                        <Alert severity="error" sx={{ mb: 2 }}>
                            {editUserError}
                        </Alert>
                    )}
                    <Box sx={{ mt: 2, display: 'flex', flexDirection: 'column', gap: 2 }}>
                        <TextField
                            label="Hourly Rate ($)"
                            name="hourly_rate"
                            type="number"
                            value={editUser.hourly_rate}
                            onChange={handleEditUserChange}
                            fullWidth
                            error={!!editFormErrors.hourly_rate}
                            helperText={editFormErrors.hourly_rate}
                            disabled={editUserLoading}
                            inputProps={{ step: "0.01", min: "0" }}
                        />

                        <FormControl fullWidth disabled={editUserLoading}>
                            <InputLabel id="edit-qualified-subtask-types-label">
                                Qualified Subtask Types
                            </InputLabel>
                            <Select
                                labelId="edit-qualified-subtask-types-label"
                                multiple
                                value={editUser.qualified_subtask_types}
                                onChange={handleEditSubtaskTypesChange}
                                input={<OutlinedInput label="Qualified Subtask Types" />}
                                renderValue={(selected) => (
                                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                                        {selected.map((value) => (
                                            <Chip key={value} label={value} />
                                        ))}
                                    </Box>
                                )}
                            >
                                {subtaskTypes.map((type) => (
                                    <MenuItem key={type.subtask_type} value={type.subtask_type}>
                                        <Checkbox checked={editUser.qualified_subtask_types.indexOf(type.subtask_type) > -1} />
                                        <ListItemText primary={type.subtask_type} />
                                    </MenuItem>
                                ))}
                            </Select>
                            <FormHelperText>Select all subtask types this user is qualified to work on</FormHelperText>
                        </FormControl>
                    </Box>
                </DialogContent>
                <DialogActions>
                    <Button onClick={handleCloseEditUserDialog} disabled={editUserLoading}>
                        Cancel
                    </Button>
                    <Button
                        onClick={handleUpdateUser}
                        variant="contained"
                        disabled={editUserLoading}
                        startIcon={editUserLoading ? <CircularProgress size={20} /> : null}
                    >
                        {editUserLoading ? 'Updating...' : 'Update User'}
                    </Button>
                </DialogActions>
            </Dialog>

            {/* Delete User Confirmation Dialog */}
            <Dialog open={openDeleteDialog} onClose={handleCloseDeleteDialog}>
                <DialogTitle>Delete User</DialogTitle>
                <DialogContent>
                    {deleteUserError && (
                        <Alert severity="error" sx={{ mb: 2 }}>
                            {deleteUserError}
                        </Alert>
                    )}
                    <Typography>
                        Are you sure you want to delete user <strong>{deleteUserId}</strong>? This action cannot be undone.
                    </Typography>
                </DialogContent>
                <DialogActions>
                    <Button onClick={handleCloseDeleteDialog} disabled={deleteUserLoading}>
                        Cancel
                    </Button>
                    <Button
                        onClick={handleDeleteUser}
                        variant="contained"
                        color="error"
                        disabled={deleteUserLoading}
                        startIcon={deleteUserLoading ? <CircularProgress size={20} /> : null}
                    >
                        {deleteUserLoading ? 'Deleting...' : 'Delete'}
                    </Button>
                </DialogActions>
            </Dialog>
        </Paper>
    );
} 