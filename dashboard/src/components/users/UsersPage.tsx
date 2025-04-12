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
    Stack
} from '@mui/material';
import { User } from '@/app/api/users/route';

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

    useEffect(() => {
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
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {users.length === 0 ? (
                            <TableRow>
                                <TableCell colSpan={columns.length} align="center">
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
        </Paper>
    );
} 