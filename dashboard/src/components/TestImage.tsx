'use client';

import Image from 'next/image';
import { Box } from '@mui/material';

export default function TestImage() {
    return (
        <Box sx={{ p: 4 }}>
            <h1>Testing Image</h1>
            <Image
                src="/ZettaLogo.png"
                alt="Zetta Logo"
                width={200}
                height={50}
                priority
            />

            <div style={{ marginTop: '20px' }}>
                <h2>Regular img tag test:</h2>
                <img
                    src="/ZettaLogo.png"
                    alt="Zetta Logo (regular img)"
                    width={200}
                    height={50}
                />
            </div>
        </Box>
    );
} 